#pragma once

#include <graph/Interaction.h>

struct RestirGBuffer
{
	SurfaceInteraction primaryInteraction;
	glm::uvec2 prevPixelCoord;

	__forceinline__ __device__ __host__ RestirGBuffer() :
		primaryInteraction({}),
		prevPixelCoord(0)
	{
	}

	__forceinline__ __device__ __host__ RestirGBuffer(
		const SurfaceInteraction& _primaryInteraction,
		const glm::uvec2& _prevPixelCoord) 
		:
		primaryInteraction(_primaryInteraction),
		prevPixelCoord(_prevPixelCoord)
	{
	}
};
