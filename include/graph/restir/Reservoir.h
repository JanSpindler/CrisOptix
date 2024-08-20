#pragma once

#include <util/random.h>
#include <graph/luminance.h>

static constexpr float CONFIDENCE_MAX = 20.0f;

template <typename T>
struct Reservoir
{
	T sample;
	float wSum;
	float confidence;

	__forceinline__ __host__ __device__ Reservoir() :
		sample(),
		wSum(0.0f),
		confidence(0.0f)
	{
	}

	__forceinline__ __host__ __device__ bool Update(const T& _sample, const float risWeight, PCG32& rng)
	{
		wSum += risWeight;
		
		confidence += 1.0f;
		confidence = glm::min(confidence, CONFIDENCE_MAX);

		if (rng.NextFloat() < risWeight / wSum)
		{
			sample = _sample;
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
			sample = inRes.sample;
			return true;
		}

		return false;
	}

	__forceinline__ __host__ __device__ bool Merge(const T& sampleTgtDom, const float inConfidence, float risWeight, PCG32& rng)
	{
		if (glm::isnan(risWeight) || glm::isinf(risWeight)) { risWeight = 0.0f; }

		wSum += risWeight;

		confidence += inConfidence;
		confidence = glm::min(confidence, CONFIDENCE_MAX);

		if (rng.NextFloat() < risWeight / wSum)
		{
			sample = sampleTgtDom;
			return true;
		}

		return false;
	}
};
