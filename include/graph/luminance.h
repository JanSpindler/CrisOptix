#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

static constexpr __forceinline__ __host__ __device__ float GetLuminance(const glm::vec3& color)
{
	return glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}
