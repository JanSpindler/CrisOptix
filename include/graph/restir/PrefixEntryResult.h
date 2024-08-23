#pragma once

#include <graph/LaunchParams.h>

struct PrefixEntryResult
{
	// Pixel idx of searching pixel
	uint32_t pixelIdx;

	// Number of found neighbors.
	uint32_t neighCount;

	// Distance to furthest neighbor.
	float maxNeighDist;

	// Index of furthest neighbor without offset.
	uint32_t maxDistNeighIdx;

	__forceinline__ __device__ __host__ PrefixEntryResult(const uint32_t _pixelIdx) :
		pixelIdx(_pixelIdx),
		neighCount(0),
		maxNeighDist(0.0f),
		maxDistNeighIdx(0)
	{
	}

	__forceinline__ __device__ void FindLargestDist(const LaunchParams& params)
	{
		// Exit if no neighbors
		if (neighCount == 0) { return; }

		//
		const uint32_t k = params.restir.gatherM - 1;
		const uint32_t offset = pixelIdx * k;

		// Get current pos
		const glm::vec3& currPos = params.restir.prefixReservoirs[pixelIdx].sample.lastInteraction.pos;

		// Go over all stored neighbors
		maxDistNeighIdx = 0;
		maxNeighDist = params.restir.prefixNeighPixels[offset + maxDistNeighIdx];
		for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
		{
			// Get neigh pos
			const uint32_t neighPixelIdx = params.restir.prefixNeighPixels[offset + neighIdx];
			const glm::vec3& neighPos = params.restir.prefixReservoirs[neighPixelIdx].sample.lastInteraction.pos;

			// Calc neigh distance
			const float dist = glm::distance(currPos, neighPos);

			// Store as max distance if it is
			if (dist < maxNeighDist)
			{
				maxDistNeighIdx = neighIdx;
				maxNeighDist = dist;
			}
		}
	}
};
