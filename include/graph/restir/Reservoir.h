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

	constexpr __forceinline__ __host__ __device__ void Reset()
	{
		wSum = 0.0f;
		confidence = 0.0f;
	}

	__forceinline__ __host__ __device__ bool Update(const T& _sample, float risWeight, PCG32& rng)
	{
		if (glm::isnan(risWeight) || glm::isinf(risWeight)) { risWeight = 0.0f; }

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

	constexpr __forceinline__ __host__ __device__ void FinalizeRIS()
	{
		constexpr float pHat = GetLuminance(sample.f);
		if (pHat == 0.0f || M == 0.0f) { wSum = 0.0f; }
		else { wSum /= pHat * M; }
	}

	constexpr __forceinline__ __host__ __device__ void FinalizeGRIS()
	{
		const float pHat = GetLuminance(sample.f);
		if (pHat == 0.0f) { wSum = 0.0f; }
		else { wSum /= pHat; }
	}
};
