#pragma once

#include <cuda.h>
#include <cstdint>
#include <cuda_runtime.h>

template <typename T>
struct CuBufferView
{
#ifdef __CUDACC__
	CUdeviceptr data;
	uint32_t count;
	uint16_t byteStride;
	uint16_t elemByteSize;
#else
	CUdeviceptr data = 0;
	uint32_t count = 0;
	uint16_t byteStride = sizeof(T);
	uint16_t elemByteSize = sizeof(T);
#endif
	
	CuBufferView() = default;

	CuBufferView(
		CUdeviceptr devicePtr,
		const uint32_t count,
		const uint16_t byteStride = sizeof(T),
		const uint16_t elemByteSize = sizeof(T))
		:
		data(devicePtr),
		count(count),
		byteStride(byteStride),
		elemByteSize(elemByteSize)
	{
	}

	__forceinline__ __host__ __device__ constexpr bool IsValid() const
	{
		return data != 0;
	}

	__forceinline__ __host__ __device__ constexpr operator bool() const
	{
		return IsValid();
	}

	__forceinline__ __host__ __device__ T& Get(uint32_t idx, const char* file = "", const int line = -1) const
	{
		if (idx >= count)
		{
			printf("Illegal memory access (%s:%d): %d >= %d\n", file, line, idx, count);
			idx = 0;
		}
		return *reinterpret_cast<T*>(data + idx * byteStride);
	}

	__forceinline__ __host__ __device__ const T& operator[](const uint32_t idx) const
	{
		return Get(idx);
	}

	__forceinline__ __host__ __device__ T& operator[](const uint32_t idx)
	{
		return Get(idx);
	}
};
