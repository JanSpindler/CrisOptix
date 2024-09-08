#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <model/Mesh.h>
#include <config.h>

static constexpr __forceinline__ __device__ glm::vec3 PointObjectToWorld(const glm::vec3& point, const glm::mat4& transform)
{
	return glm::vec3(transform * glm::vec4(point, 1.0f));
}

static constexpr __forceinline__ __device__ glm::vec3 NormalObjectToWorld(const glm::vec3& normal, const glm::mat4& transform)
{
	return glm::vec3(transform * glm::vec4(normal, 0.0f));
}

template <typename T>
static constexpr __forceinline__ __device__ T InterpolateBary(const glm::vec2& baryCoord, const T& v0, const T& v1, const T& v2)
{
	return
		((1.0f - baryCoord.x - baryCoord.y) * v0)
		+ (baryCoord.x * v1)
		+ (baryCoord.y * v2);
}

struct HitInfo
{
	const MeshSbtData* meshSbtData;
	uint32_t instanceId;
	uint32_t primitiveIdx;
	Vec2 baryCoord;

	__forceinline__ __device__ __host__ HitInfo() :
		meshSbtData(nullptr),
		instanceId(0),
		primitiveIdx(0),
		baryCoord(0.0f)
	{
	}

	constexpr __forceinline__ __device__ __host__ HitInfo(
		const MeshSbtData* _meshSbtData,
		const uint32_t _instanceId,
		const uint32_t _primitiveIdx,
		const Vec2& _baryCoord)
		:
		meshSbtData(_meshSbtData),
		instanceId(_instanceId),
		primitiveIdx(_primitiveIdx),
		baryCoord(_baryCoord)
	{
	}
};

struct PackedInteraction
{
	HitInfo hitInfo;
	Vec3 inRayDir;

	__forceinline__ __device__ __host__ PackedInteraction() :
		hitInfo({}),
		inRayDir(0.0f)
	{
	}

	constexpr __forceinline__ __device__ __host__ PackedInteraction(const HitInfo& _hitInfo, const Vec3& _inRayDir) :
		hitInfo(_hitInfo),
		inRayDir(_inRayDir)
	{
	}

	constexpr __forceinline__ __device__ __host__ bool IsValid() const
	{
		return hitInfo.meshSbtData != nullptr;
	}
};

struct Interaction
{
	const MeshSbtData* meshSbtData;
	uint32_t instanceId;
	uint32_t primitiveIdx;
	glm::vec3 pos;
	glm::vec3 inRayDir;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	glm::vec2 baryCoord;
	bool valid;

	__forceinline__ __host__ __device__ Interaction() :
		valid(false),
		inRayDir(0.0f),
		pos(0.0f),
		normal(0.0f),
		tangent(0.0f),
		uv(0.0f),
		instanceId(0),
		primitiveIdx(0),
		baryCoord(0.0f),
		meshSbtData(nullptr)
	{
	}

	__forceinline__ __device__ Interaction(const PackedInteraction& packedInt, const CuBufferView<glm::mat4>& transforms)
		:
		valid(packedInt.IsValid()),
		inRayDir(packedInt.inRayDir),
		pos(0.0f),
		normal(0.0f),
		tangent(0.0f),
		uv(0.0f),
		instanceId(packedInt.hitInfo.instanceId),
		primitiveIdx(packedInt.hitInfo.primitiveIdx),
		baryCoord(packedInt.hitInfo.baryCoord),
		meshSbtData(packedInt.hitInfo.meshSbtData)
	{
		// Exit if not called from kernel
#ifndef __CUDACC__
		valid = false;
		return;
#endif

		// Exit if mesh data invalid
		if (meshSbtData == nullptr)
		{
			valid = false;
			return;
		}

		// Get transform
		// TODO: Fix index
		const glm::mat4& glmTransform = transforms[0];

		// Indices of triangle vertices in the mesh
		const glm::uvec3 indices(
			meshSbtData->indices[primitiveIdx * 3 + 0],
			meshSbtData->indices[primitiveIdx * 3 + 1],
			meshSbtData->indices[primitiveIdx * 3 + 2]);

		// Vertices
		const Vertex& v0 = meshSbtData->vertices[indices.x];
		const Vertex& v1 = meshSbtData->vertices[indices.y];
		const Vertex& v2 = meshSbtData->vertices[indices.z];

		// Interpolate
		pos = InterpolateBary<glm::vec3>(baryCoord, v0.pos, v1.pos, v2.pos);
		pos = PointObjectToWorld(pos, glmTransform);

		normal = InterpolateBary<glm::vec3>(baryCoord, v0.normal, v1.normal, v2.normal);
		normal = NormalObjectToWorld(normal, glmTransform);

		tangent = InterpolateBary<glm::vec3>(baryCoord, v0.tangent, v1.tangent, v2.tangent);
		tangent = NormalObjectToWorld(tangent, glmTransform);

		uv = InterpolateBary<glm::vec2>(baryCoord, v0.uv, v1.uv, v2.uv);
	}

	constexpr __forceinline__ __device__ __host__ operator PackedInteraction() const
	{
		const HitInfo hitInfo(valid ? meshSbtData : nullptr, instanceId, primitiveIdx, baryCoord);
		return PackedInteraction(hitInfo, inRayDir);
	}
};

//struct InteractionSeed
//{
//	glm::vec3 pos;
//	Vec3 inDir;
//
//	__forceinline__ __device__ __host__ InteractionSeed() :
//		pos(0.0f),
//		inDir(0.0f)
//	{
//	}
//
//	constexpr __forceinline__ __device__ __host__ InteractionSeed(const glm::vec3& _pos, const Vec3& _inDir) :
//		pos(_pos),
//		inDir(_inDir)
//	{
//	}
//
//	__forceinline__ __device__ __host__ InteractionSeed(const Interaction& interaction) :
//		pos(interaction.pos),
//		inDir(interaction.inRayDir)
//	{
//	}
//};
