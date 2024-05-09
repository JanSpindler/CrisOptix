#pragma once

#include <cuda.h>
#include <cstdint>
#include <cuda_runtime.h>

template <typename T>
struct CuBufferView
{
	CUdeviceptr data = 0;
	uint32_t count = 0;
	uint16_t byteStride = sizeof(T);
	uint16_t elemByteSize = sizeof(T);

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

	__host__ __device__ constexpr bool IsValid() const
	{
		return data != 0;
	}

	__host__ __device__ constexpr operator bool() const
	{
		return IsValid();
	}

	__host__ __device__ const T& operator[](const uint32_t idx) const
	{
		return *reinterpret_cast<T*>(data + idx * byteStride);
	}

	__host__ __device__ T& operator[](const uint32_t idx)
	{
		return *reinterpret_cast<T*>(data + idx * byteStride);
	}
};
