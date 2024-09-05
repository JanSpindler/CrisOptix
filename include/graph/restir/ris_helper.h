#pragma once

#include <cuda_runtime.h>
#include <cuda/std/tuple>

static constexpr __forceinline__ __device__ __host__ float CalcUnbiasedContributionWeightWy(const float pHat, const float wSum)
{
	return wSum / pHat;
}

static constexpr __forceinline__ __device__ __host__ float CalcResamplingWeightWi(
	const float misWeight, 
	const float pHatTgtDom, 
	const float ucwSrcDom, 
	const float jacobian)
{
	return glm::max(0.0f, misWeight * pHatTgtDom * ucwSrcDom * jacobian);
}
