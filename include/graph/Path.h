#pragma once

#include <cuda/std/array>
#include <glm/glm.hpp>
#include <model/Emitter.h>

static constexpr size_t MAX_PATH_LEN = 8; // 0 = direct illumination
static constexpr size_t MAX_PATH_RAND_COUNT = 3; // 3 for BRDF sampling and 5 for emitter sampling

struct Path
{
	union RandVar
	{
		uint32_t randUInt;
		float randFloat;
	};

	cuda::std::array<glm::vec3, MAX_PATH_LEN> vertices;
	cuda::std::array<cuda::std::array<RandVar, MAX_PATH_RAND_COUNT>, MAX_PATH_LEN> randomVars;
	glm::vec3 throughput;
	glm::vec3 outputRadiance;
	size_t prefixLength;
	size_t length;
	EmitterSample emitterSample;
};
