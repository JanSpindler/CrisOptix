#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>

struct PrefixPath
{
	// Random number generator state before prefix was generated
	PCG32 rng;

	// Primary interaction
	InteractionSeed primaryIntSeed;

	// Interaction at first vertex fit for prefix reconnection
	InteractionSeed reconIntSeed;

	// Interaction at last vertex (used for generating and reconnection with suffix)
	InteractionSeed lastIntSeed;

	// Path flags
	// 0:8 -> length: Vertex count starting at primary hit. Does not include NEE hit.
	// 8:16 -> reconnection index: Index of first vertex fit for prefix reconnection
	// 16:17 -> valid: True if prefix is a valid path
	// 17:18 -> nee: True if prefix was terminated early by NEE
	uint32_t flags;
	
	// Throughput or radiance if nee hit.
	glm::vec3 f;

	// Throughput or radiance after the reconnection vertex
	glm::vec3 postReconF;

	// Sampling pdf
	float p;

	__forceinline__ __device__ __host__ PrefixPath() :
		flags(0u),
		primaryIntSeed({}),
		reconIntSeed({}),
		rng({}),
		f(0.0f),
		p(0.0f),
		lastIntSeed({}),
		postReconF(0.0f)
	{
	}

	__forceinline__ __device__ __host__ PrefixPath(
		const PrefixPath& other,
		const glm::vec3& _f,
		const float _p,
		const InteractionSeed& _primaryIntSeed)
		:
		flags(other.flags),
		primaryIntSeed(_primaryIntSeed),
		reconIntSeed(other.reconIntSeed),
		rng(other.rng),
		f(_f),
		postReconF(other.postReconF),
		p(_p),
		lastIntSeed(other.lastIntSeed)
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
		return static_cast<bool>(flags & (1 << 16u));
	}

	constexpr __forceinline__ __device__ __host__ void SetValid(const bool valid)
	{
		flags &= ~(1 << 16u);
		if (valid) { flags |= 1 << 16u; }
	}

	constexpr __forceinline__ __device__ __host__ bool IsNee() const
	{
		return static_cast<bool>(flags & (1 << 17u));
	}

	constexpr __forceinline__ __device__ __host__ void SetNee(const bool nee)
	{
		flags &= ~(1 << 17u);
		if (nee) { flags |= 1 << 17u; }
	}
};
