#pragma once

#include <cuda/std/array>
#include <glm/glm.hpp>

static constexpr size_t MAX_PATH_LEN = 8; // 0 = direct illumination
static constexpr size_t MAX_PATH_RAND_COUNT = 5; // 3 for BRDF sampling and 5 for emitter sampling

struct Path
{
	cuda::std::array<glm::vec3, MAX_PATH_LEN> vertices;
	cuda::std::array<cuda::std::array<float, MAX_PATH_RAND_COUNT>, MAX_PATH_LEN> randomVars;
	glm::vec3 outputRadiance;
};
