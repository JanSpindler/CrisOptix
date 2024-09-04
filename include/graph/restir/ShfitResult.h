#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct ShiftResult
{
	glm::vec3 shiftedF;
	float jacobian;

	__forceinline__ __device__ __host__ ShiftResult() :
		shiftedF(0.0f),
		jacobian(0.0f)
	{
	}

	__forceinline__ __device__ __host__ ShiftResult(const glm::vec3& _shiftedF, const float _jacobian) :
		shiftedF(_shiftedF),
		jacobian(_jacobian)
	{
	}
};
