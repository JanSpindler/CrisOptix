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
	T currentSample;
	float wSum;
	size_t M;
	float currentIntegrand;

	constexpr __host__ __device__ void Update(const T& sample, const float weight, const float integrand, PCG32& rng)
	{
		wSum += weight;
		++M;
		if (rng.NextFloat() < weight / wSum)
		{
			currentSample = sample;
			currentIntegrand = integrand;
		}
	}

	constexpr __host__ __device__ bool MergeSameDomain(const Reservoir<T>& inRes, const float misWeight, PCG32& rng)
	{
		float weight = inRes.integrand * inRes.wSum * misWeight;
		if (std::isnan(weight) || std::isinf(weight)) { weight = 0.0f; }

		M += inRes.M;
		wSum += weight;

		if (rng.NextFloat() < weight / wSum)
		{
			currentSample = inRes.currentSample;
			currentIntegrand = inRes.currentIntegrand;
			return true;
		}

		return false;
	}

	constexpr __host__ __device__ bool Merge(const Reservoir<T>& inRes, const float integrand, const float jacobian, const float misWeight, PCG32& rng)
	{
		float weight = integrand * jacobian * inRes.wSum * misWeight;
		if (std::isnan(weight) || std::isinf(weight)) { weight = 0.0f; }

		M += inRes.M;
		wSum = weight;

		if (rng.NextFloat() < weight / wSum)
		{
			currentSample = inRes.currentSample;
			currentIntegrand = integrand;
			return true;
		}

		return false;
	}
};
