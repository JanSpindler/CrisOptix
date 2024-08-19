#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <util/pixel_index.h>
#include <graph/restir/struct/Reservoir.h>
#include <graph/restir/struct/SuffixPath.h>

static __forceinline__ __device__ void TemporalSuffixReuse(
	const size_t pixelIdx,
	const glm::uvec2& prevPixelCoord,
	const PrefixPath& prefix,
	Reservoir<SuffixPath>& suffixRes,
	Reconnection& recon,
	PCG32& rng,
	const LaunchParams& params)
{
	// Exit if prev pixel is invalid
	if (!IsPixelValid(prevPixelCoord, params)) { return; }
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);

	// Get prev suffix reservoir
	const Reservoir<SuffixPath>& prevSuffixRes = params.restir.suffixReservoirs[prevPixelIdx];
	const SuffixPath& prevSuffix = prevSuffixRes.currentSample;

	// Exit if prev suffix is invalid
	if (!prevSuffix.valid) { return; }
	
	//
	Reconnection prevRecon{};
	if (!CalcReconnection(prevRecon, prefix, prevSuffix, params)) { return; };

	// Calc jacobian
	const float jacobian = GetReconnectionJacobian(
		suffixRes.currentSample.firstPos, 
		prevSuffix.firstPos, 
		prefix.lastInteraction.pos,
		prefix.lastInteraction.normal);

	// Stream temportal reuse suffix into res
	// TODO: Store reconnection in SuffixPath because here we need the contribution in neigh domain
	const float pairwiseK = 1.0f;
	const float misWeight = CompNeighPairwiseMisWeight(
		prevRecon.GetWeight3f() * prevSuffix.throughput, 
		prevSuffix.throughput, 
		jacobian, 
		pairwiseK, 
		suffixRes.M, 
		prevSuffixRes.M);
	if (suffixRes.Merge(prevSuffixRes, prevSuffix.throughput, 1.0f, misWeight, rng))
	{
		recon = prevRecon;
	}
}
