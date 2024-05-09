#pragma once

#include <cuda.h>
#include <cstdint>

template <typename T>
struct CuBufferView
{
	CUdeviceptr data = 0;
	uint32_t count = 0;
	uint16_t byteStride = sizeof(T);
	uint16_t elemByteSize = sizeof(T);

	CuBufferView() = default;

	CuBufferView(
		const CUdeviceptr devicePtr,
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

	constexpr bool IsValid() const
	{
		return data != 0;
	}

	constexpr operator bool() const
	{
		return IsValid();
	}

	const T& operator[](const uint32_t idx) const
	{
		return *reinterpret_cast<T*>(data + idx * byteStride);
	}

	T& operator[](const uint32_t idx)
	{
		return *reinterpret_cast<T*>(data + idx * byteStride);
	}
};
