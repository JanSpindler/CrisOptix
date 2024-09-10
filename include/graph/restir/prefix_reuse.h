#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/trace.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/ris_helper.h>
#include <graph/restir/ShfitResult.h>

// Shifts current prefix into others domain and evaluates the f of the old domain
static __forceinline__ __device__ glm::vec3 CalcCurrContribInOtherDomain(
	const PrefixPath& currPrefix,
	const PrefixPath& otherPrefix,
	float& jacobian, // Jacobian: Shift current into other
	const LaunchParams& params)
{
	// Default jacobian
	jacobian = 0.0f;

	// Get current primary interaction
	const Interaction currPrimaryInt(currPrefix.primaryInt, params.transforms);
	if (!currPrimaryInt.valid) { return; }

	// Get other primary interaction
	Interaction otherPrimaryInt(otherPrefix.primaryInt, params.transforms);
	if (!otherPrimaryInt.valid) { return glm::vec3(0.0f); }

	// Get reconnection interaction
	Interaction currReconInt(currPrefix.reconInt, params.transforms);
	if (!currReconInt.valid) { return glm::vec3(0.0f); }

	// Hybrid shift
	const uint32_t reconVertCount = glm::max<int>(otherPrefix.GetReconIdx() - 2, 0);
	Interaction& currInt = otherPrimaryInt;
	PCG32 otherRng = otherPrefix.rng;
	glm::vec3 throughput(1.0f);
	for (uint32_t idx = 0; idx < reconVertCount; ++idx)
	{
		// Sampled brdf
		const BrdfSampleResult brdf = optixDirectCall<BrdfSampleResult, const Interaction&, PCG32&>(
			currInt.meshSbtData->sampleMaterialSbtIdx,
			currInt,
			otherRng);
		if (brdf.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
		throughput *= brdf.brdfVal;

		// Recon dir
		const glm::vec3& reconDir = brdf.outDir;
		
		// Trace new interaction
		const glm::vec3 oldPos = currInt.pos;
		TraceWithDataPointer<Interaction>(params.traversableHandle, oldPos, reconDir, 1e-3f, 1e16f, params.surfaceTraceParams, currInt);
		if (!currInt.valid) { return glm::vec3(0.0f); }

		// Trace occlusion
		if (TraceOcclusion(oldPos, reconDir, 1e-3f, glm::distance(oldPos, currInt.pos), params)) { return glm::vec3(0.0f); }
	}

	// Final reconnection segment
	// Check occlusion
	const glm::vec3 reconVec = currReconInt.pos - currInt.pos;
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

	// Fix recon interaction in dir
	currReconInt.inRayDir = reconDir;

	// Brdf eval 2
	const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currReconInt.meshSbtData->evalMaterialSbtIdx,
		currReconInt,
		currPrefix.reconOutDir);
	if (brdf2.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }
	throughput *= brdf2.brdfResult;

	// Jacobian inverse shift
	jacobian = CalcReconnectionJacobian(
		currPrimaryInt.pos,
		otherPrimaryInt.pos,
		currReconInt.pos,
		currReconInt.normal);

	// Inverse shifted p hat
	const glm::vec3 shiftedF = throughput * currPrefix.postReconF;

	//
	return shiftedF;
}
