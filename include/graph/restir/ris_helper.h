#pragma once

#include <cuda_runtime.h>
#include <cuda/std/tuple>

static __forceinline__ __device__ __host__ float CalcUnbiasedContributionWeightWy(const float pHat, const float wSum)
{
	return wSum / pHat;
}

static __forceinline__ __device__ __host__ float CalcResamplingWeightWi(
	const float misWeight, 
	const float pHatTgtDom, 
	const float ucwSrcDom, 
	const float jacobian)
{
	return misWeight * pHatTgtDom * ucwSrcDom * jacobian;
}

static __forceinline__ __device__ __host__ float CalcPFromI(const float pHatSrcDom, const float invJacobian)
{
	return pHatSrcDom * invJacobian;
}

static __forceinline__ __device__ __host__ cuda::std::pair<float, float> CalcTalbotMisWeightsMi(
	const float pFromI0, 
	const float confidence0,
	const float pFromI1,
	const float confidence1)
{
	const float m0 = pFromI0 * confidence0;
	const float m1 = pFromI1 * confidence1;
	const float sum = m0 + m1;
	return { m0 / sum, m1 / sum };
}
