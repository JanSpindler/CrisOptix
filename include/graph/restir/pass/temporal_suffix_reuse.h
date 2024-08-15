#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <util/pixel_index.h>
#include <graph/restir/struct/Reservoir.h>
#include <graph/restir/struct/SuffixPath.h>

static __forceinline__ __device__ void TemporalSuffixReuse(
	const size_t pixelIdx,
	const glm::uvec2& prevPixelCoord,
	const LaunchParams& params)
{
	// Exit if prev pixel is invalid
	if (IsPixelValid(prevPixelCoord, params)) { return; }
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);

	// Get prev suffix reservoir
	const Reservoir<SuffixPath>& prevSuffixRes = params.restir.suffixReservoirs[prevPixelIdx];
}
