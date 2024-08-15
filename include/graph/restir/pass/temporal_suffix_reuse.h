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
	printf("Temp suffix reuse\n");

	//
	Reconnection prevRecon{};
	if (!CalcReconnection(prevRecon, prefix, prevSuffix, params)) { return; };

	// Stream temportal reuse suffix into res
	const float jacobian = 1.0f; // TODO: Calc jacobian in CalcReconnection()
	// TODO: Check if mis weight is correct
	if (suffixRes.Merge(prevSuffixRes, prevSuffix.throughput, 1.0f, prevRecon.GetP(), rng))
	{
		//printf("Temp suffix reuse\n");
		recon = prevRecon;
	}
}
