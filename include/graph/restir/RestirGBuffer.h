#pragma once

#include <graph/Interaction.h>

struct RestirGBuffer
{
	SurfaceInteraction primaryInteraction;

	__forceinline__ __device__ __host__ RestirGBuffer() :
		primaryInteraction({})
	{
	}

	__forceinline__ __device__ __host__ RestirGBuffer(const SurfaceInteraction& _primaryInteraction) :
		primaryInteraction(_primaryInteraction)
	{
	}
};
