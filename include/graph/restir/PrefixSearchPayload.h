#pragma once

#include <graph/LaunchParams.h>

struct PrefixSearchPayload
{
	// Pixel idx of searching pixel
	uint32_t pixelIdx;

	// Number of found neighbors.
	uint32_t neighCount;

	// Distance to furthest neighbor.
	float maxNeighDist;

	// Index of furthest neighbor without offset.
	uint32_t maxDistNeighIdx;

	// Intersection count
	uint32_t intersectionCount;

	__forceinline__ __device__ __host__ PrefixSearchPayload(const uint32_t _pixelIdx) :
		pixelIdx(_pixelIdx),
		neighCount(0),
		maxNeighDist(0.0f),
		maxDistNeighIdx(0),
		intersectionCount(0)
	{
	}

	constexpr __forceinline__ __device__ const PrefixNeighbor& GetNeighbor(const uint32_t neighIdx, const LaunchParams& params) const
	{
		const uint32_t k = params.restir.gatherM - 1;
		const uint32_t offset = pixelIdx * k;
		const PrefixNeighbor& neigh = params.restir.prefixNeighbors[offset + neighIdx];
		return neigh;
	}

	__forceinline__ __device__ void FindLargestDist(const LaunchParams& params)
	{
		return;

		// Exit if no neighbors
		if (neighCount == 0) { return; }

		//
		const uint32_t k = params.restir.gatherM - 1;
		const uint32_t offset = pixelIdx * k;

		// Get last prefix interaction
		const Interaction lastPrefixInt(
			params.restir.prefixReservoirs[2 * pixelIdx + params.restir.frontBufferIdx].sample.lastInt, 
			params.transforms);
		if (!lastPrefixInt.IsValid()) { return; }

		// Get current pos
		const glm::vec3& currPos = lastPrefixInt.pos;

		// Go over all stored neighbors
		maxDistNeighIdx = 0;
		maxNeighDist = GetNeighbor(maxDistNeighIdx, params).distance;

		for (uint32_t neighIdx = 0; neighIdx < neighCount; ++neighIdx)
		{
			// Get neigh
			const PrefixNeighbor& neigh = GetNeighbor(neighIdx, params);

			// Store as max distance if it is
			if (neigh.distance < maxNeighDist)
			{
				maxDistNeighIdx = neighIdx;
				maxNeighDist = neigh.distance;
			}
		}
	}
};
