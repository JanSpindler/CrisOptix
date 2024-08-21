#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/restir/ris_helper.h>

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
	if (otherSuffix.reconIdx > 1) { return; }

	// Exit if occluded
	const glm::vec3 reconDir = glm::normalize(otherSuffix.reconInteraction.pos - prefix.lastInteraction.pos);
	const float reconDist = glm::distance(otherSuffix.reconInteraction.pos, prefix.lastInteraction.pos);
	const bool occluded = TraceOcclusion(
		params.traversableHandle,
		prefix.lastInteraction.pos,
		reconDir,
		1e-3f,
		reconDist,
		params.occlusionTraceParams);
	if (occluded) { return; }

	// Calc mis weights
	const float jacobian = CalcReconnectionJacobian(
		otherSuffix.lastPrefixPos,
		currSuffix.lastPrefixPos,
		otherSuffix.reconInteraction.pos,
		otherSuffix.reconInteraction.normal);
	const float pFromCurr = GetLuminance(currSuffix.f);
	const float pFromPrev = CalcPFromI(GetLuminance(otherSuffix.f), 1.0f / jacobian);
	const cuda::std::pair<float, float> misWeights = CalcTalbotMisWeightsMi(pFromCurr, currRes.confidence, pFromPrev, otherRes.confidence);

	// Shift prefix path to target domain
	// TODO: Add hybrid shift
	// Evaluate brdf at primary interaction towards reconnection vertex
	const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
		prefix.lastInteraction.meshSbtData->evalMaterialSbtIdx,
		prefix.lastInteraction,
		reconDir);

	// Calc shifted f and p
	const glm::vec3 shiftedF = brdfEvalResult.brdfResult * otherSuffix.postReconF;
	const float shiftedP = 
		params.neeProb * // Sampled NEE at some point
		brdfEvalResult.samplingPdf * // BRDF sample at reconnection
		glm::pow(1.0f - params.neeProb, static_cast<float>(otherSuffix.len)); // BRDF sample for path after reconnection

	// Construct shifted PrefixPath
	const SuffixPath shiftedSuffix = SuffixPath(otherSuffix, currSuffix.lastPrefixPos, currSuffix.lastPrefixInDir, shiftedF, shiftedP);
	
	// Calc ris weight
	const float risWeight = CalcResamplingWeightWi(misWeights.second, GetLuminance(shiftedF), otherRes.wSum, jacobian);

	// Merge reservoirs
	if (currRes.Merge(shiftedSuffix, otherRes.confidence, risWeight, rng))
	{
		// Only for debug purposes
		printf("hi");
	}
}
