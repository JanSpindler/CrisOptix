#pragma once

#include <cstddef>

template <typename T>
class DeviceBuffer
{
public:
	constexpr DeviceBuffer(size_t count = 0)
	{
		Alloc(count);
	}

	constexpr ~DeviceBuffer()
	{
		Free();
	}

private:
	size_t m_Count = 0;
	size_t m_AllocCount = 0;
	T* m_Ptr = nullptr;
};
