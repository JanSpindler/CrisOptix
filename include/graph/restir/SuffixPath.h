#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>
#include <graph/luminance.h>
#include <config.h>

struct SuffixPath
{
	// State of random number generator before tracing suffix
	PCG32 rng;

	// Interaction at recon vertex
	PackedInteraction reconInt;

	// Last prefix interaction
	PackedInteraction lastPrefixInt;

	// Path flags
	// 0:8 -> length: Length of path without NEE vertex (0 meaning direct termination into NEE)
	// 8:16 -> reconnection index: Index of recon vertex
	// 16:17 -> valid
	uint32_t flags;

	// Path contribution after prefix to light source (including prefix-suffix connection)
	glm::vec3 f;

	// Path contribution after reconnection vertex
	glm::vec3 postReconF;

	// Sampling pdf
	float p;

	// Out direction after reconnection (used for brdf evaluation)
	Vec3 reconOutDir;

	__forceinline__ __device__ __host__ SuffixPath() :
		flags(0),
		lastPrefixInt({}),
		reconInt({}),
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
		const PackedInteraction& _lastPrefixInt,
		const glm::vec3& _f)
		:
		flags(other.flags),
		lastPrefixInt(_lastPrefixInt),
		reconInt(other.reconInt),
		reconOutDir(other.reconOutDir),
		f(_f),
		postReconF(other.postReconF),
		p(other.p),
		rng(other.rng)
	{
	}

	constexpr __forceinline__ __device__ __host__ void Reset()
	{
		flags = 0;
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
