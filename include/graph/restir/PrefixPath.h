#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>

struct PrefixPath
{
	// True if prefix is a valid path
	bool valid;

	// True if prefix was terminated early by NEE
	bool nee;

	// Interaction at first vertex fit for prefix reconnection
	SurfaceInteraction reconInteraction;

	// Index of first vertex fit for prefix reconnection
	uint32_t reconIdx;

	// Random number generator state before prefix was generated
	PCG32 rng;

	// Throughput or radiance if nee hit.
	glm::vec3 f;

	// Sampling pdf
	float p;

	// Vertex count starting at primary hit. Does not include NEE hit.
	uint32_t len;

	__forceinline__ __device__ __host__ PrefixPath() :
		valid(false),
		nee(false),
		reconInteraction({}),
		reconIdx(0),
		rng({}),
		f(0.0f),
		p(0.0f),
		len(0)
	{
	}
};
