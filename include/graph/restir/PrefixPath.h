#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>

struct PrefixPath
{
	// True if prefix is a valid path
	bool valid;

	// True if prefix was terminated early by NEE
	bool nee;

	// Position of primary hit
	glm::vec3 primaryHitPos;

	// Direction leading into primary hit
	glm::vec3 primaryHitInDir;

	// Interaction at first vertex fit for prefix reconnection
	SurfaceInteraction reconInteraction;

	// Index of first vertex fit for prefix reconnection
	uint32_t reconIdx;

	// Random number generator state before prefix was generated
	PCG32 rng;

	// Throughput or radiance if nee hit.
	glm::vec3 f;

	// Throughput or radiance after the reconnection vertex
	glm::vec3 postReconF;

	// Sampling pdf
	float p;

	// Vertex count starting at primary hit. Does not include NEE hit.
	uint32_t len;

	__forceinline__ __device__ __host__ PrefixPath() :
		valid(false),
		nee(false),
		primaryHitPos(0.0f),
		primaryHitInDir(0.0f),
		reconInteraction({}),
		reconIdx(0),
		rng({}),
		f(0.0f),
		p(0.0f),
		len(0)
	{
	}

	__forceinline__ __device__ __host__ PrefixPath(
		const PrefixPath& other,
		const glm::vec3& _f,
		const float _p,
		const glm::vec3& _primaryHitPos,
		const glm::vec3& _primaryHitInDir)
		:
		valid(other.valid),
		nee(other.nee),
		primaryHitPos(_primaryHitPos),
		primaryHitInDir(_primaryHitInDir),
		reconInteraction(other.reconInteraction),
		reconIdx(other.reconIdx),
		rng(other.rng),
		f(_f),
		postReconF(other.postReconF),
		p(_p),
		len(other.len)
	{
	}
};
