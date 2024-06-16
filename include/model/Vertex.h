#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;

	__host__ __device__ Vertex() :
		pos(0.0f),
		normal(0.0f),
		tangent(0.0f),
		uv(0.0f)
	{
	}

	__host__ __device__ Vertex(const glm::vec3& pos, const glm::vec3& normal, const glm::vec3& tangent, const glm::vec2& uv) :
		pos(pos),
		normal(normal),
		tangent(tangent),
		uv(uv)
	{
	}
};
