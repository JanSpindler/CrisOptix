#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>
#include <graph/luminance.h>

struct SuffixPath
{
	// Valid flag
	bool valid;

	// Position of last prefix vertex (used for jacobian)
	glm::vec3 lastPrefixPos;

	// In dir at last prefix vertex (used for jacobian)
	glm::vec3 lastPrefixInDir;

	// Interaction at recon vertex
	SurfaceInteraction reconInteraction;

	// Index of recon vertex
	uint32_t reconIdx;

	// Path contribution after prefix to light source (including prefix-suffix connection)
	glm::vec3 f;

	// Path contribution after reconnection vertex
	glm::vec3 postReconF;

	// Sampling pdf
	float p;

	// Length of path without NEE vertex (0 meaning direct termination into NEE)
	uint32_t len;

	// State of random number generator before tracing suffix
	PCG32 rng;

	__forceinline__ __device__ __host__ SuffixPath() :
		valid(false),
		lastPrefixPos(0.0f),
		lastPrefixInDir(0.0f),
		reconInteraction({}),
		reconIdx(0),
		f(0.0f),
		postReconF(0.0f),
		p(0.0f),
		len(0),
		rng({})
	{
	}

	// Constructor used for shift mapping
	__forceinline__ __device__ __host__ SuffixPath(
		const SuffixPath& other,
		const glm::vec3& _lastPrefixPos,
		const glm::vec3& _lastPrefixInDir,
		const glm::vec3& _f,
		const float _p)
		:
		valid(other.valid),
		lastPrefixPos(_lastPrefixPos),
		lastPrefixInDir(_lastPrefixInDir),
		reconInteraction(other.reconInteraction),
		reconIdx(other.reconIdx),
		f(_f),
		postReconF(other.postReconF),
		p(_p),
		len(other.len),
		rng(other.rng)
	{
	}
};
