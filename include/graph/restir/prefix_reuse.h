#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/trace.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/ShfitResult.h>

// Shifts source prefix into destination domain and evaluates the f of the old domain
static __forceinline__ __device__ glm::vec3 CalcCurrContribInOtherDomain(
	const PrefixPath& srcPrefix,
	const PrefixPath& dstPrefix,
	float& jacobian, // Jacobian: Shift current into other
	const LaunchParams& params)
{
	// Default jacobian
	jacobian = 0.0f;

	//
	if (!srcPrefix.IsValid() || !srcPrefix.IsValid() || srcPrefix.GetReconIdx() < 2) { return glm::vec3(0.0f); }

	// Get current primary interaction
	const Interaction srcPrimaryInt(srcPrefix.primaryInt, params.transforms);
	if (!srcPrimaryInt.valid) { return glm::vec3(0.0f); }

	// Get other primary interaction
	Interaction dstPrimaryInt(dstPrefix.primaryInt, params.transforms);
	if (!dstPrimaryInt.valid) { return glm::vec3(0.0f); }

	// Get reconnection interaction
	Interaction srcReconInt(srcPrefix.reconInt, params.transforms);
	if (!srcReconInt.valid) { return glm::vec3(0.0f); }

	// Hybrid shift
	const uint32_t reconVertCount = glm::max<int>(srcPrefix.GetReconIdx() - 2, 0);
	Interaction& currInt = dstPrimaryInt;
	PCG32 otherRng = srcPrefix.rng;
	glm::vec3 throughput(1.0f);
	for (uint32_t idx = 0; idx < reconVertCount; ++idx)
	{
		//printf("%d, %d\n", otherPrefix.GetLength(), otherPrefix.GetReconIdx());

		// Sampled brdf
		const BrdfSampleResult brdf = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			currInt.meshSbtData->sampleMaterialSbtIdx,
			currInt,
			otherRng);
		if (brdf.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
		throughput *= brdf.brdfVal;

		// Trace new interaction
		TraceWithDataPointer<Interaction>(params.traversableHandle, currInt.pos, brdf.outDir, 1e-3f, 1e16f, params.surfaceTraceParams, currInt);
		if (!currInt.valid) { return glm::vec3(0.0f); }
	}

	// Final reconnection segment
	// Check occlusion
	const glm::vec3 reconVec = srcReconInt.pos - currInt.pos;
	const float reconLen = glm::length(reconVec);
	const glm::vec3 reconDir = glm::normalize(reconVec);

	if (glm::isinf(reconLen) || glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return glm::vec3(0.0f); }
	if (TraceOcclusion(currInt.pos, reconDir, 1e-3f, reconLen, params)) { return glm::vec3(0.0f); }

	// Brdf eval 1
	const BrdfEvalResult brdf1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currInt.meshSbtData->evalMaterialSbtIdx,
		currInt,
		reconDir);
	if (brdf1.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
	throughput *= brdf1.brdfResult;

	// Fix recon interaction in dir
	srcReconInt.inRayDir = reconDir;

	// Brdf eval 2
	const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		srcReconInt.meshSbtData->evalMaterialSbtIdx,
		srcReconInt,
		srcPrefix.reconOutDir);
	if (brdf2.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
	throughput *= brdf2.brdfResult;

	// Jacobian inverse shift
	jacobian = CalcReconnectionJacobian(
		srcPrimaryInt.pos,
		currInt.pos,
		srcReconInt.pos,
		srcReconInt.normal);

	// Inverse shifted p hat
	const glm::vec3 shiftedF = glm::max(glm::vec3(0.0f), throughput * srcPrefix.postReconF);

	//
	return shiftedF;
}
