#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>

static __forceinline__ __device__ void SuffixReuse(
	Reservoir<SuffixPath>& currRes,
	const Reservoir<SuffixPath>& otherRes,
	PCG32& rng,
	const LaunchParams& params)
{
	// Assume: Both reservoirs and samples are valid

	// Get prefixes from res
	const SuffixPath& currSuffix = currRes.sample;
	const SuffixPath& otherSuffix = otherRes.sample;

	// Reconnect at first vertex (0 for first vertex by nee)
	// TODO: Support hybrid shift
	if (otherSuffix.reconIdx > 1) { return; }

	// Exit if occluded
	const glm::vec3 reconDir = glm::normalize(otherPrefix.reconInteraction.pos - primaryInteraction.pos);
	const float reconDist = glm::distance(otherPrefix.reconInteraction.pos, primaryInteraction.pos);
	const bool occluded = TraceOcclusion(
		params.traversableHandle,
		primaryInteraction.pos,
		reconDir,
		1e-3f,
		reconDist,
		params.occlusionTraceParams);
	if (occluded) { return; }

	// Calc mis weights
	const float jacobian = CalcReconnectionJacobian(
		otherPrefix.primaryHitPos,
		primaryInteraction.pos,
		otherPrefix.reconInteraction.pos,
		otherPrefix.reconInteraction.normal);
	const float pFromCurr = GetLuminance(currPrefix.f);
	const float pFromPrev = CalcPFromI(GetLuminance(otherPrefix.f), 1.0f / jacobian);
	const cuda::std::pair<float, float> misWeights = CalcTalbotMisWeightsMi(pFromCurr, res.confidence, pFromPrev, otherRes.confidence);

	// Shift prefix path to target domain
	// TODO: Add hybrid shift
	// Evaluate brdf at primary interaction towards reconnection vertex
	const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
		primaryInteraction.meshSbtData->evalMaterialSbtIdx,
		primaryInteraction,
		reconDir);

	// Calc shifted f and p
	const glm::vec3 shiftedF = brdfEvalResult.brdfResult * otherPrefix.postReconF;
	const float shiftedP = brdfEvalResult.samplingPdf * glm::pow(1.0f - params.neeProb, static_cast<float>(otherPrefix.len - 1));

	// Construct shifted PrefixPath
	const PrefixPath shiftedPrefix = PrefixPath(otherPrefix, shiftedF, shiftedP, primaryInteraction.pos, primaryInteraction.inRayDir);

	// Calc ris weight
	const float risWeight = CalcResamplingWeightWi(misWeights.second, GetLuminance(shiftedF), otherRes.wSum, jacobian);

	// Merge reservoirs
	if (res.Merge(shiftedPrefix, otherRes.confidence, risWeight, rng))
	{
		// Only for debug purposes
		//printf("hi");
	}
}
