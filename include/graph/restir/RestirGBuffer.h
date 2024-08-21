#pragma once

#include <graph/Interaction.h>

struct RestirGBuffer
{
	SurfaceInteraction primaryInteraction;
	glm::uvec2 prevPixelCoord;
	PCG32 rng;

	__forceinline__ __device__ __host__ RestirGBuffer() :
		primaryInteraction({}),
		prevPixelCoord(0),
		rng({})
	{
	}

	__forceinline__ __device__ __host__ RestirGBuffer(
		const SurfaceInteraction& _primaryInteraction,
		const glm::uvec2& _prevPixelCoord,
		const PCG32& _rng)
		:
		primaryInteraction(_primaryInteraction),
		prevPixelCoord(_prevPixelCoord),
		rng(_rng)
	{
	}
};
