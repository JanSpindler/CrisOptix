#pragma once

#include <util/random.h>

struct RestirParams
{
	int canonicalCount;

	bool enableTemporal;

	bool enableSpatial;
	int spatialCount;
	int spatialKernelSize;

	uint32_t prefixLength;
};

template <typename T>
struct Reservoir
{
	T y;
	float wSum;
	size_t M;
	float W;

	constexpr __host__ __device__ void Update(const T& sample, const float weight, PCG32& rng)
	{
		wSum += weight;
		++M;
		if (rng.NextFloat() < weight / wSum)
		{
			y = sample;
		}
	}
};
