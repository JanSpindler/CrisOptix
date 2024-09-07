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
	// Get other primary interaction
	Interaction otherPrimaryInt{};
	TraceInteractionSeed(otherPrefix.primaryIntSeed, otherPrimaryInt, params);
	if (!otherPrimaryInt.valid) { return glm::vec3(0.0f); }

	// Get reconnection interaction
	Interaction currReconInt{};
	TraceInteractionSeed(currPrefix.reconIntSeed, currReconInt, params);
	if (!currReconInt.valid) { return glm::vec3(0.0f); }

	// Jacobian inverse shift
	jacobian = CalcReconnectionJacobian(
		currPrefix.primaryIntSeed.pos,
		otherPrimaryInt.pos,
		currReconInt.pos,
		currReconInt.normal);

	// Check occlusion
	const glm::vec3 reconVec = currReconInt.pos - otherPrimaryInt.pos;
	const float reconLen = glm::length(reconVec);
	const glm::vec3 reconDir = glm::normalize(reconVec);

	if (glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return glm::vec3(0.0f); }
	if (TraceOcclusion(otherPrimaryInt.pos, reconDir, 1e-3f, reconLen, params)) { return glm::vec3(0.0f); }

	// Fix recon interaction in dir
	currReconInt.inRayDir = reconDir;

	// Brdf eval 1
	const BrdfEvalResult brdf1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		otherPrimaryInt.meshSbtData->evalMaterialSbtIdx,
		otherPrimaryInt,
		reconDir);
	if (brdf1.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }

	// Brdf eval 2
	const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currReconInt.meshSbtData->evalMaterialSbtIdx,
		currReconInt,
		currPrefix.reconOutDir);
	if (brdf2.samplingPdf <= 0.0f) { return glm::vec3(0.0f); }

	// Inverse shifted p hat
	const glm::vec3 shiftedF = brdf1.brdfResult * brdf2.brdfResult * currPrefix.postReconF;

	//
	return shiftedF;
}
