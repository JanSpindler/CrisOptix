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
	glm::vec3 fOverP;

	__forceinline__ __host__ __device__ Reservoir() :
		sample(),
		wSum(0.0f),
		confidence(0.0f),
		fOverP(0.0f)
	{
	}

	__forceinline__ __host__ __device__ bool Update(const T& _sample, const float risWeight, const glm::vec3& _fOverP, PCG32& rng)
	{
		wSum += risWeight;
		
		confidence += 1.0f;
		confidence = glm::min(confidence, CONFIDENCE_MAX);

		if (rng.NextFloat() < risWeight / wSum)
		{
			sample = _sample;
			fOverP = _fOverP;
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
			fOverP = inRes.fOverP;
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
			sample = inRes.sample;
			fOverP = inRes.fOverP;
			return true;
		}

		return false;
	}
};
