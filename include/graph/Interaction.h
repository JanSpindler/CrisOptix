#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <model/Mesh.h>

struct Interaction
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

	__forceinline__ __host__ __device__ Interaction() :
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

struct InteractionSeed
{
	glm::vec3 pos;
	glm::vec3 inDir;

	__forceinline__ __device__ __host__ InteractionSeed() :
		pos(0.0f),
		inDir(0.0f)
	{
	}

	constexpr __forceinline__ __device__ __host__ InteractionSeed(const glm::vec3& _pos, const glm::vec3& _inDir) :
		pos(_pos),
		inDir(_inDir)
	{
	}

	constexpr __forceinline__ __device__ __host__ InteractionSeed(const Interaction& interaction) :
		pos(interaction.pos),
		inDir(interaction.inRayDir)
	{
	}
};
