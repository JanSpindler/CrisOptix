#pragma once

#include <util/random.h>

struct LightSample
{

};

template <typename T>
struct Reservoir
{
	T outputSample;
	float weightSum;
	size_t sampleCount;

	__host__ __device__ constexpr void Update(const T& sample, const float weight, PCG32& rng)
	{
		weightSum += weight;
		++sampleCount;
		if (rng.NextFloat() < weight / weightSum) { outputSample = sample; }
	}
};
