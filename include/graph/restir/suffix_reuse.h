#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/restir/ris_helper.h>
#include <graph/restir/path_gen.h>
#include <graph/trace.h>

static __forceinline__ __device__ void SuffixReuse(
	Reservoir<SuffixPath>& currRes,
	const Reservoir<SuffixPath>& otherRes,
	const PrefixPath& prefix,
	PCG32& rng,
	const LaunchParams& params)
{
	// Assume: Both reservoirs and samples are valid
	// Assume: Prefix is valid
	// Assume: currRes.sample.lastPrefixPos == prefix.lastInteraction.pos

	// Get prefixes from res
	const SuffixPath& currSuffix = currRes.sample;
	const SuffixPath& otherSuffix = otherRes.sample;

	// Reconnect at first vertex (0 for first vertex by nee / 1 for normal surface hit)
	// TODO: Support hybrid shift
	if (otherSuffix.GetReconIdx() > 1) { return; }

	// Calc recon vector
	const glm::vec3 reconDir = glm::normalize(otherSuffix.reconIntSeed.pos - prefix.lastIntSeed.pos);
	const float reconDist = glm::distance(otherSuffix.reconIntSeed.pos, prefix.lastIntSeed.pos);

	// Exit if reconnection vector is invalid
	if (glm::isinf(reconDist) || glm::isnan(reconDist) || reconDist < 1e-2f) { return; }

	// Check occlusion
	const bool occluded = TraceOcclusion(
		prefix.lastIntSeed.pos,
		reconDir,
		1e-3f,
		reconDist,
		params);
	if (occluded) { return; }

	// Get reconnection interaction from seed
	Interaction reconInteraction{};
	TraceInteractionSeed(otherSuffix.reconIntSeed, reconInteraction, params);
	if (!reconInteraction.valid) { return; }

	// Calc mis weights
	const float jacobian = CalcReconnectionJacobian(
		otherSuffix.lastPrefixIntSeed.pos,
		currSuffix.lastPrefixIntSeed.pos,
		reconInteraction.pos,
		reconInteraction.normal);

	// Get last prefix interaction
	Interaction lastPrefixInt{};
	TraceInteractionSeed(prefix.lastIntSeed, lastPrefixInt, params);
	if (!lastPrefixInt.valid) { return; }

	// Shift prefix path to target domain
	// TODO: Add hybrid shift
	// Evaluate brdf at primary interaction towards reconnection vertex
	const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const Interaction&, const glm::vec3&>(
		lastPrefixInt.meshSbtData->evalMaterialSbtIdx,
		lastPrefixInt,
		reconDir);

	// Calc shifted f and p
	const glm::vec3 shiftedF = brdfEvalResult.brdfResult * otherSuffix.postReconF;
	const float shiftedP = 
		params.neeProb * // Sampled NEE at some point
		//brdfEvalResult.samplingPdf * // BRDF sample at reconnection
		glm::pow(1.0f - params.neeProb, static_cast<float>(otherSuffix.GetLength())); // BRDF sample for path after reconnection
	
	// Construct shifted PrefixPath
	const SuffixPath shiftedSuffix = SuffixPath(otherSuffix, currSuffix.lastPrefixIntSeed, shiftedF, shiftedP);
	
	// Calc ris weight
	// TODO
	const float risWeight = CalcResamplingWeightWi(1.0f, GetLuminance(shiftedF), otherRes.wSum, jacobian);

	// Merge reservoirs
	if (currRes.Merge(shiftedSuffix, otherRes.confidence, risWeight, rng))
	{
		// Only for debug purposes
		//printf("hi");
	}
}
