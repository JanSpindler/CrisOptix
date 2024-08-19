#pragma once

#include <util/random.h>
#include <graph/luminance.h>

static constexpr float CONFIDENCE_MAX = 20.0f;

template <typename T>
struct Reservoir
{
	T currentSample;
	float wSum;
	float confidence;
	glm::vec3 currentIntegrand; // f / p

	__forceinline__ __host__ __device__ Reservoir() :
		wSum(0.0f),
		confidence(0.0f),
		currentIntegrand(0.0f)
	{
	}

	__forceinline__ __host__ __device__ Reservoir(const T& sample) :
		T(sample),
		wSum(0.0f),
		confidence(0.0f),
		currentIntegrand(0.0f)
	{
	}

	__forceinline__ __host__ __device__ bool Update(const T& sample, const float risWeight, const glm::vec3& integrand, PCG32& rng)
	{
		wSum += risWeight;
		
		confidence += 1.0f;
		confidence = glm::min(confidence, CONFIDENCE_MAX);

		if (rng.NextFloat() < risWeight / wSum)
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
		if (glm::isnan(weight) || glm::isinf(weight)) { weight = 0.0f; }

		wSum += weight;

		confidence += inRes.confidence;
		confidence = glm::min(confidence, CONFIDENCE_MAX);

		if (rng.NextFloat() < weight / wSum)
		{
			currentSample = inRes.currentSample;
			currentIntegrand = inRes.currentIntegrand;
			return true;
		}

		return false;
	}

	__forceinline__ __host__ __device__ bool Merge(
		const Reservoir<T>& inRes, 
		const glm::vec3& integrand, 
		const float jacobian, 
		const float misWeight, 
		PCG32& rng)
	{
		float weight = GetLuminance(integrand) * jacobian * inRes.wSum * misWeight;
		if (glm::isnan(weight) || glm::isinf(weight)) { weight = 0.0f; }

		wSum = weight;

		confidence += inRes.confidence;
		confidence = glm::min(confidence, CONFIDENCE_MAX);

		if (rng.NextFloat() < weight / wSum)
		{
			currentSample = inRes.currentSample;
			currentIntegrand = integrand;
			return true;
		}

		return false;
	}
};
