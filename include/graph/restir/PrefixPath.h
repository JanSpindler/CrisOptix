#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>

struct PrefixPath
{
	bool valid;
	bool nee;

	SurfaceInteraction lastInteraction;
	
	glm::vec3 throughput;
	float p;

	uint32_t len;

	__forceinline__ __device__ __host__ PrefixPath() :
		valid(false),
		nee(false),
		lastInteraction({}),
		throughput(0.0f),
		p(0.0f),
		len(0)
	{
	}
};
