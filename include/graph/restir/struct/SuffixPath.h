#pragma once

#include <glm/glm.hpp>
#include <util/random.h>

struct SuffixPath
{
	bool valid;

	glm::vec3 firstPos;

	glm::vec3 radiance;
	float p;

	uint32_t len;

	PCG32 rng;

	__forceinline__ __device__ float GetWeight() const
	{
		return GetLuminance(radiance) / p;
	}
};
