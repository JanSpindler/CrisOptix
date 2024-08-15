#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>
#include <graph/luminance.h>

struct SuffixPath
{
	bool valid;

	glm::vec3 firstPos;
	glm::vec3 firstDir;

	glm::vec3 radiance;
	float p;

	uint32_t len;

	PCG32 rng;

	__forceinline__ __device__ __host__ SuffixPath() :
		valid(false),
		firstPos(0.0f),
		firstDir(0.0f),
		radiance(0.0f),
		p(0.0f),
		len(0),
		rng({})
	{
	}

	__forceinline__ __device__ float GetWeight() const
	{
		return GetLuminance(radiance) / p;
	}
};
