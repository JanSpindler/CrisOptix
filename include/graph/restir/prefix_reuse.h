#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/trace.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/ris_helper.h>
#include <cuda/std/tuple>

static __forceinline__ __device__ void ShiftPrefix(
	const PrefixPath& currPrefix,
	const PrefixPath& otherPrefix)
{
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

	// Get primary interaction
	Interaction primaryInteraction{};
	TraceInteractionSeed(currPrefix.primaryIntSeed, primaryInteraction, params);
	if (!primaryInteraction.valid) { return; }

	// Exit if prev prefix res is invalid or unfit for reuse
	if (!otherPrefix.IsValid() || otherPrefix.GetLength() < params.restir.minPrefixLen) { return; }

	// Reconnect after primary hit
	// TODO: Support hybrid shift
	if (otherPrefix.GetReconIdx() != 2) { return; }

	// Exit if occluded
	const glm::vec3 reconDir = glm::normalize(otherPrefix.reconIntSeed.pos - primaryInteraction.pos);
	const float reconDist = glm::distance(otherPrefix.reconIntSeed.pos, primaryInteraction.pos);
	const bool occluded = TraceOcclusion(
		primaryInteraction.pos,
		reconDir,
		1e-3f,
		reconDist,
		params);
	if (occluded) { return; }

	// Get other prefix recon interaction
	Interaction otherPrefixReconInt{};
	TraceInteractionSeed(otherPrefix.reconIntSeed, otherPrefixReconInt, params);
	if (!otherPrefixReconInt.valid) { return; }

	// Calc mis weights
	const float jacobian = CalcReconnectionJacobian(
		otherPrefix.primaryIntSeed.pos,
		primaryInteraction.pos,
		otherPrefixReconInt.pos,
		otherPrefixReconInt.normal);
	const float pFromCurr = GetLuminance(currPrefix.f);
	const float pFromPrev = CalcPFromI(GetLuminance(otherPrefix.f), 1.0f / jacobian);
	const cuda::std::pair<float, float> misWeights = CalcTalbotMisWeightsMi(pFromCurr, res.confidence, pFromPrev, otherRes.confidence);

	// Shift prefix path to target domain
	// TODO: Add hybrid shift
	// Evaluate brdf at primary interaction towards reconnection vertex
	const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		primaryInteraction.meshSbtData->evalMaterialSbtIdx,
		primaryInteraction,
		reconDir);

	// Calc shifted f and p
	const glm::vec3 shiftedF = brdfEvalResult.brdfResult * otherPrefix.postReconF;
	const float shiftedP = 
		//brdfEvalResult.samplingPdf * 
		glm::pow(1.0f - params.neeProb, static_cast<float>(otherPrefix.GetLength() - 1));

	// Construct shifted PrefixPath
	const PrefixPath shiftedPrefix = PrefixPath(otherPrefix, shiftedF, shiftedP, primaryInteraction);

	// Calc ris weight
	const float risWeight = CalcResamplingWeightWi(misWeights.second, GetLuminance(shiftedF), otherRes.wSum, jacobian);

	// Merge reservoirs
	if (res.Merge(shiftedPrefix, otherRes.confidence, risWeight, rng))
	{
		// Only for debug purposes
		//printf("hi");
	}
}
