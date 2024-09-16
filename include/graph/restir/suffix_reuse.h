#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <graph/trace.h>
#include <graph/restir/ShfitResult.h>

// Shifts sourc prefix into destination domain and evaluates the target function "p hat from i"
static __forceinline__ __device__ glm::vec3 CalcCurrContribInOtherDomain(
	const SuffixPath& srcSuffix,
	const Interaction& dstLastPrefixInt,
	float& jacobian, // Jacobian: Shift current into other
	const LaunchParams& params)
{
	// Default val jacobian
	jacobian = 0.0f;

	// Check source suffix
	if (!srcSuffix.IsValid()) { return glm::vec3(0.0f); }

	// Get other last perfix interaction
	if (!dstLastPrefixInt.valid) { return glm::vec3(0.0f); }

	// Get current last prefix interaction
	const Interaction srcLastPrefixInt(srcSuffix.lastPrefixInt, params.transforms);
	if (!srcLastPrefixInt.valid) { return glm::vec3(0.0f); }

	// Get reconnection interaction
	Interaction srcReconInt(srcSuffix.reconInt, params.transforms);
	if (!srcReconInt.valid) { return glm::vec3(0.0f); }

	// Hybrid shift
	const uint32_t reconVertCount = glm::max<int>(srcSuffix.GetReconIdx() - 1, 0);
	Interaction currInt = dstLastPrefixInt;
	PCG32 srcRng = srcSuffix.rng;
	glm::vec3 throughput(1.0f);
	for (uint32_t idx = 0; idx < reconVertCount; ++idx)
	{
		//printf("%d\n", otherSuffix.GetReconIdx());

		// Dummy rng because of NEE
		srcRng.NextFloat();

		// Sampled brdf
		const BrdfSampleResult brdf = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			currInt.meshSbtData->sampleMaterialSbtIdx,
			currInt,
			srcRng);
		if (brdf.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
		throughput *= brdf.brdfVal;

		// Recon dir
		const glm::vec3& reconDir = brdf.outDir;

		// Trace new interaction
		const glm::vec3 oldPos = currInt.pos;
		TraceWithDataPointer<Interaction>(params.traversableHandle, oldPos, reconDir, 1e-3f, 1e16f, params.surfaceTraceParams, currInt);
		if (!currInt.valid) { return glm::vec3(0.0f); }
	}

	// Check occlusion
	const glm::vec3 reconVec = srcReconInt.pos - currInt.pos;
	const float reconLen = glm::length(reconVec);
	const glm::vec3 reconDir = glm::normalize(reconVec);

	if (glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return glm::vec3(0.0f); }
	if (TraceOcclusion(currInt.pos, reconDir, 1e-3f, reconLen, params)) { return glm::vec3(0.0f); }

	// Brdf eval 1
	const BrdfEvalResult brdf1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currInt.meshSbtData->evalMaterialSbtIdx,
		currInt,
		reconDir);
	if (brdf1.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
	throughput *= brdf1.brdfResult;

	// Brdf eval 2
	if (srcSuffix.GetReconIdx() < srcSuffix.GetLength())
	{
		// Fix recon interaction in dir
		srcReconInt.inRayDir = reconDir;

		//
		if (srcSuffix.reconOutDir == glm::vec3(0.0f)) { return glm::vec3(0.0f); }

		// Brdf
		const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
			srcReconInt.meshSbtData->evalMaterialSbtIdx,
			srcReconInt,
			srcSuffix.reconOutDir);
		if (brdf2.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
		throughput *= brdf2.brdfResult;
	}

	// Jacobian inverse shift
	jacobian = CalcReconnectionJacobian(
		srcLastPrefixInt.pos,
		currInt.pos,
		srcReconInt.pos,
		srcReconInt.normal);

	// Inverse shift p hat
	const glm::vec3 shiftedF = throughput * srcSuffix.postReconF;

	//
	return shiftedF;
}

// Shifts source suffix into destination domain and evaluates the target function "p hat from i"
static __forceinline__ __device__ glm::vec3 CalcCurrContribInOtherDomain(
	const SuffixPath& srcSuffix,
	const SuffixPath& dstSuffix,
	float& jacobian, // Jacobian: Shift current into other
	const LaunchParams& params)
{
	// Get other last perfix interaction
	Interaction dstLastPrefixInt(dstSuffix.lastPrefixInt, params.transforms);
	if (!dstLastPrefixInt.valid) { return glm::vec3(0.0f); }

	//
	return CalcCurrContribInOtherDomain(srcSuffix, dstLastPrefixInt, jacobian, params);
}
