#pragma once

#include <util/random.h>

struct RestirParams
{
	bool enableTemporal;
	bool enableSpatial;
	int canonicalCount;
	int spatialCount;
	int spatialKernelSize;
};

template <typename T>
struct Reservoir
{
	T y;
	float wSum;
	size_t M;
	float W;

	__host__ __device__ constexpr void Update(const T& sample, const float weight, PCG32& rng)
	{
		wSum += weight;
		++M;
		if (rng.NextFloat() < weight / wSum)
		{
			y = sample;
		}
	}
};
