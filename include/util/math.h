#pragma once

#include <cuda_runtime.h>

template <typename T>
static constexpr __forceinline__ __host__ __device__ T CeilToMultiple(const T value, const T mod)
{
	return value - T(1) + mod - ((value - T(1)) % mod);
}

template <typename T>
static constexpr __forceinline__ __host__ __device__ T CeilDiv(const T num, const T den)
{
	return (num - T(1)) / den + T(1);
}
