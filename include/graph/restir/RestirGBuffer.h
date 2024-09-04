#pragma once

#include <graph/Interaction.h>

struct RestirGBuffer
{
	glm::uvec2 prevPixelCoord;
	PCG32 rng;

	__forceinline__ __device__ __host__ RestirGBuffer() :
		prevPixelCoord(0),
		rng({})
	{
	}

	__forceinline__ __device__ __host__ RestirGBuffer(
		const glm::uvec2& _prevPixelCoord,
		const PCG32& _rng)
		:
		prevPixelCoord(_prevPixelCoord),
		rng(_rng)
	{
	}
};
