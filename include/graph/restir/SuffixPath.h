#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>
#include <graph/luminance.h>

struct SuffixPath
{
	// Valid flag
	bool valid;

	// Interaction at recon vertex
	SurfaceInteraction reconInteraction;

	// Index of recon vertex
	uint32_t reconIdx;

	// Path contribution after prefix to light source (including prefix-suffix connection)
	glm::vec3 f;

	// Path contribution after reconnection vertex
	glm::vec3 postReconF;

	// Sampling pdf
	float p;

	// Length of path without NEE vertex (0 meaning direct termination into NEE)
	uint32_t len;

	// State of random number generator before tracing suffix
	PCG32 rng;

	__forceinline__ __device__ __host__ SuffixPath() :
		valid(false),
		reconInteraction({}),
		reconIdx(0),
		f(0.0f),
		postReconF(0.0f),
		p(0.0f),
		len(0),
		rng({})
	{
	}
};
