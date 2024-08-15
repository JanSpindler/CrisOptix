#pragma once

#include <optix_device.h>
#include <graph/restir/struct/PathReservoir.h>
#include <graph/LaunchParams.h>
#include <graph/Interaction.h>
#include <graph/trace.h>
#include <util/pixel_index.h>
#include <glm/glm.hpp>

static __forceinline__ __device__ void PrefixProduceRetrace(
	const size_t pixelIdx,
	const glm::uvec2& prevPixelCoord,
	const PathReservoir& centralRes,
	const SurfaceInteraction& primaryInteraction,
	LaunchParams& params)
{
	// Exit if interaction is invalid
	if (!primaryInteraction.valid) { return; }

	// Temporal
	// Stop if prev pixel is not on screen
	if (!IsPixelValid(prevPixelCoord, params)) { return; }
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);

	//
	const SurfaceInteraction& prevInteraction = params.restir.primaryInteractions[prevPixelIdx];
	if (prevInteraction.valid)
	{
		// ?
		uint32_t numWorks = 0;
		uint32_t workMask = 0;

		static constexpr int startReplayPrefixLength = 1;

		const PathReservoir& prevRes = params.restir.pathReservoirs[prevPixelIdx];

		int p1 = centralRes.pathFlags.PrefixLength() > startReplayPrefixLength &&
			centralRes.pathFlags.PathTreeLength() >= centralRes.pathFlags.PrefixLength();		
		numWorks += p1;
		workMask |= p1;
		
		int p2 = int(prevRes.pathFlags.PrefixLength() > startReplayPrefixLength &&
			prevRes.pathFlags.PathTreeLength() >= prevRes.pathFlags.PrefixLength());
		numWorks += p2;
		workMask |= p2 << 1;
	}
}
