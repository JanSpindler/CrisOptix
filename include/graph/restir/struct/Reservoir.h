#pragma once

#include <util/random.h>

template <typename T>
struct Reservoir
{
	T currentSample;
	float wSum;
	size_t M;
	glm::vec3 currentIntegrand; // f / p

	__forceinline__ __host__ __device__ Reservoir() :
		wSum(0.0f),
		M(0),
		currentIntegrand(0.0f)
	{
	}

	__forceinline__ __host__ __device__ Reservoir(const T& sample) :
		T(sample),
		wSum(0.0f),
		M(0),
		currentIntegrand(0.0f)
	{
	}

	__forceinline__ __host__ __device__ bool Update(const T& sample, const float weight, const glm::vec3& integrand, PCG32& rng)
	{
		wSum += weight;
		++M;
		if (rng.NextFloat() < weight / wSum)
		{
			currentSample = sample;
			currentIntegrand = integrand;
			return true;
		}
		return false;
	}

	__forceinline__ __host__ __device__ bool MergeSameDomain(const Reservoir<T>& inRes, const float misWeight, PCG32& rng)
	{
		float weight = GetLuminance(inRes.integrand) * inRes.wSum * misWeight;
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

	__forceinline__ __host__ __device__ bool Merge(const Reservoir<T>& inRes, const glm::vec3& integrand, const float jacobian, const float misWeight, PCG32& rng)
	{
		float weight = GetLuminance(integrand) * jacobian * inRes.wSum * misWeight;
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
