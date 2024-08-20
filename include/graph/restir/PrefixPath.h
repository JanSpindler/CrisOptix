#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>

struct PrefixPath
{
	bool valid;
	bool nee;

	SurfaceInteraction lastInteraction;
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
		lastInteraction({}),
		rng({}),
		f(0.0f),
		p(0.0f),
		len(0)
	{
	}
};
