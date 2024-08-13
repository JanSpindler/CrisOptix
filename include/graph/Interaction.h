#pragma once

#include <glm/glm.hpp>
#include <model/Mesh.h>
#include <cuda_runtime.h>

struct SurfaceInteraction
{
	bool valid;
	glm::vec3 inRayDir;
	float inRayDist;
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	uint32_t primitiveIdx;
	const MeshSbtData* meshSbtData;

	constexpr __device__ __host__ SurfaceInteraction() :
		valid(false),
		inRayDir(0.0f),
		inRayDist(0.0f),
		pos(0.0f),
		normal(0.0f),
		tangent(0.0f),
		uv(0.0f),
		primitiveIdx(0),
		meshSbtData(nullptr)
	{
	}
};
