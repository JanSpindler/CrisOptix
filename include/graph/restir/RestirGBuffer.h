#pragma once

#include <graph/Interaction.h>

struct RestirGBuffer
{
	PCG32 rng;
	glm::uvec2 prevPixelCoord;
	bool primaryIntValid;

	__forceinline__ __device__ __host__ RestirGBuffer() :
		prevPixelCoord(0),
		rng({}),
		primaryIntValid(false)
	{
	}

	__forceinline__ __device__ __host__ RestirGBuffer(
		const glm::uvec2& _prevPixelCoord,
		const PCG32& _rng,
		const bool _primaryIntValid)
		:
		prevPixelCoord(_prevPixelCoord),
		rng(_rng),
		primaryIntValid(_primaryIntValid)
	{
	}
};
