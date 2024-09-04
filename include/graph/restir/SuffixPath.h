#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>
#include <graph/luminance.h>
#include <cuda_fp16.h>

struct SuffixPath
{
	// Interaction at recon vertex
	Interaction reconInteraction;

	// State of random number generator before tracing suffix
	PCG32 rng;

	// Path flags
	// 0:8 -> length: Length of path without NEE vertex (0 meaning direct termination into NEE)
	// 8:16 -> reconnection index: Index of recon vertex
	// 16:17 -> valid
	uint32_t flags;

	// Position of last prefix vertex (used for jacobian)
	glm::vec3 lastPrefixPos;

	// In dir at last prefix vertex (used for jacobian)
	glm::vec3 lastPrefixInDir;

	// Out direction after reconnection (used for brdf evaluation)
	glm::vec3 reconOutDir;

	// Path contribution after prefix to light source (including prefix-suffix connection)
	glm::vec3 f;

	// Path contribution after reconnection vertex
	glm::vec3 postReconF;

	// Sampling pdf
	float p;

	__forceinline__ __device__ __host__ SuffixPath() :
		flags(0),
		lastPrefixPos(0.0f),
		lastPrefixInDir(0.0f),
		reconInteraction({}),
		reconOutDir(0.0f),
		f(0.0f),
		postReconF(0.0f),
		p(0.0f),
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
		flags(other.flags),
		lastPrefixPos(_lastPrefixPos),
		lastPrefixInDir(_lastPrefixInDir),
		reconInteraction(other.reconInteraction),
		reconOutDir(other.reconOutDir),
		f(_f),
		postReconF(other.postReconF),
		p(_p),
		rng(other.rng)
	{
	}

	constexpr __forceinline__ __device__ __host__ uint32_t GetLength() const
	{
		return static_cast<uint32_t>(flags & 0xFF);
	}

	constexpr __forceinline__ __device__ __host__ void SetLength(const uint32_t length)
	{
		flags &= ~0xFFu;
		flags |= length & 0xFFu;
	}

	constexpr __forceinline__ __device__ __host__ uint32_t GetReconIdx() const
	{
		return static_cast<uint32_t>((flags >> 8u) & 0xFFu);
	}

	constexpr __forceinline__ __device__ __host__ void SetReconIdx(const uint32_t reconIdx)
	{
		flags &= ~0xFF00u;
		flags |= (reconIdx & 0xFFu) << 8u;
	}

	constexpr __forceinline__ __device__ __host__ bool IsValid() const
	{
		return static_cast<bool>(flags & (1 << 16));
	}

	constexpr __forceinline__ __device__ __host__ void SetValid(const bool valid)
	{
		flags &= ~(1 << 16);
		if (valid) { flags |= 1 << 16; }
	}
};
