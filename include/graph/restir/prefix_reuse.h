#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/trace.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/ris_helper.h>
#include <graph/restir/ShfitResult.h>

// Shifts current prefix into others domain and evaluates the target function "p hat from i"
static __forceinline__ __device__ float CalcOtherMisWeight(
	const PrefixPath& currPrefix,
	const PrefixPath& otherPrefix,
	const LaunchParams& params)
{
	// Get other primary interaction
	Interaction otherPrimaryInt{};
	TraceInteractionSeed(otherPrefix.primaryIntSeed, otherPrimaryInt, params);
	if (!otherPrimaryInt.valid) { return 0.0f; }

	// Get reconnection interaction
	Interaction currReconInt{};
	TraceInteractionSeed(currPrefix.reconIntSeed, currReconInt, params);
	if (!currReconInt.valid) { return 0.0f; }

	// Jacobian inverse shift
	const float jacobian = CalcReconnectionJacobian(
		currPrefix.primaryIntSeed.pos,
		otherPrimaryInt.pos,
		currReconInt.pos,
		currReconInt.normal);

	// Check occlusion
	const glm::vec3 reconVec = currReconInt.pos - otherPrimaryInt.pos;
	const float reconLen = glm::length(reconVec);
	const glm::vec3 reconDir = glm::normalize(reconVec);

	if (glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return 0.0f; }
	if (TraceOcclusion(otherPrimaryInt.pos, reconDir, 1e-3f, reconLen, params)) { return 0.0f; }

	// Brdf eval 1
	const BrdfEvalResult brdf1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		otherPrimaryInt.meshSbtData->evalMaterialSbtIdx,
		otherPrimaryInt,
		reconDir);
	if (brdf1.samplingPdf <= 0.0f) { return 0.0f; }

	// Brdf eval 2
	const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currReconInt.meshSbtData->evalMaterialSbtIdx,
		currReconInt,
		currPrefix.reconOutDir);
	if (brdf2.samplingPdf <= 0.0f) { return 0.0f; }

	// Inverse shifted p hat
	const glm::vec3 shiftedF = brdf1.brdfResult * brdf2.brdfResult * currPrefix.postReconF;

	//
	return GetLuminance(shiftedF) * jacobian;
}

static __forceinline__ __device__ ShiftResult ShiftPrefix(
	const PrefixPath& currPrefix,
	const PrefixPath& otherPrefix,
	const LaunchParams& params)
{
	//
	if (otherPrefix.GetReconIdx() != 2) { return ShiftResult(); }

	// Trace occlusion
	const glm::vec3 reconVec = otherPrefix.reconIntSeed.pos - currPrefix.primaryIntSeed.pos;
	const float reconLen = glm::length(reconVec);
	const glm::vec3 reconDir = glm::normalize(reconVec);

	if (glm::any(glm::isinf(reconDir) || glm::isnan(reconDir))) { return ShiftResult(); }
	if (TraceOcclusion(currPrefix.primaryIntSeed.pos, reconDir, 1e-3f, reconLen, params)) { return ShiftResult(); }

	// Get primary interaction
	Interaction currPrimaryInt{};
	TraceInteractionSeed(currPrefix.primaryIntSeed, currPrimaryInt, params);
	if (!currPrimaryInt.valid) { return ShiftResult(); }

	// Eval brdf 1
	const BrdfEvalResult brdf1 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		currPrimaryInt.meshSbtData->evalMaterialSbtIdx,
		currPrimaryInt,
		reconDir);
	if (brdf1.samplingPdf <= 0.0f) { return ShiftResult(); }

	// Get reconnection interaction
	Interaction otherReconInt{};
	TraceInteractionSeed(otherPrefix.reconIntSeed, otherReconInt, params);
	if (!otherReconInt.valid) { return ShiftResult(); }

	// Eval brdf 2
	const BrdfEvalResult brdf2 = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		otherReconInt.meshSbtData->evalMaterialSbtIdx,
		otherReconInt,
		otherPrefix.reconOutDir);
	if (brdf2.samplingPdf <= 0.0f) { return ShiftResult(); }

	// Calc shifted f
	const glm::vec3 shiftedF = brdf1.brdfResult * brdf2.brdfResult * otherPrefix.postReconF;

	// Calc jacobian
	const float jacobian = CalcReconnectionJacobian(
		otherPrefix.primaryIntSeed.pos, 
		currPrimaryInt.pos, 
		otherReconInt.pos,
		otherReconInt.normal);
	
	//
	return { shiftedF, jacobian };
}

static __forceinline__ __device__ void PrefixReuse(
	Reservoir<PrefixPath>& res,
	const Reservoir<PrefixPath>& otherRes,
	PCG32& rng,
	const LaunchParams& params)
{
	// Assume: res and res.sample are valid

	// Get prefixes from res
	const PrefixPath& currPrefix = res.sample;
	const PrefixPath& otherPrefix = otherRes.sample;

	// Exit if prev prefix res is invalid or unfit for reuse
	if (!otherPrefix.IsValid() || otherPrefix.GetLength() < params.restir.minPrefixLen) { return; }

	//
	const ShiftResult shiftResult = ShiftPrefix(currPrefix, otherPrefix, params);

	// Construct shifted PrefixPath
	const PrefixPath shiftedPrefix = PrefixPath(otherPrefix, shiftResult.shiftedF, currPrefix.primaryIntSeed);

	// MIS weight | 1 = current sample and 2 = other sample | first index = domain and second index = sample
	const float m22 = GetLuminance(otherPrefix.f) * otherRes.confidence;
	const float m12 = CalcOtherMisWeight(currPrefix, otherPrefix, params) * res.confidence;
	const float misWeight = m22 / (m12 + m22);

	// Calc ris weight
	const float risWeight = CalcResamplingWeightWi(
		misWeight, 
		GetLuminance(shiftResult.shiftedF), 
		otherRes.wSum / GetLuminance(otherPrefix.f), 
		shiftResult.jacobian);

	// Merge reservoirs
	if (res.Merge(shiftedPrefix, otherRes.confidence, risWeight, rng))
	{
		// Only for debug purposes
		//printf("%f\n", risWeight);
	}
}
