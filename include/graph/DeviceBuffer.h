#pragma once

#include <cstddef>
#include <util/custom_assert.h>
#include <cuda_runtime.h>

template <typename T>
class DeviceBuffer
{
public:
	constexpr DeviceBuffer(const size_t count = 0)
	{
		Alloc(count);
	}

	~DeviceBuffer()
	{
		Free();
	}

	constexpr void Alloc(const size_t count)
	{
		Free();
		m_AllocCount = count;
		m_Count = count;
		if (m_Count > 0)
		{
			ASSERT_CUDA(cudaMalloc(&m_Ptr, m_AllocCount * sizeof(T)));
		}
	}

	constexpr void AllocIfRequired(const size_t count)
	{
		if (count <= m_Count)
		{
			m_Count = count;
			return;
		}
		Alloc(count);
	}

	constexpr void Free()
	{
		m_Count = 0;
		m_AllocCount = 0;
		ASSERT_CUDA(cudaFree(m_Ptr));
		m_Ptr = nullptr;
	}

	constexpr void UploadSub(const T* data, const size_t offset, const size_t count)
	{
		if (offset + count > m_AllocCount)
		{
			Log::Error("Uploaded too much to DeviceBuffer<>", true);
		}
		ASSERT_CUDA(cudaMemcpy(m_Ptr + offset, data, count * sizeof(T), cudaMemcpyHostToDevice));
	}

	constexpr void Upload(const T* data)
	{
		UploadSub(data, 0, m_Count);
	}

	constexpr void DownloadSub(T* data, const size_t offset, const size_t count) const
	{
		if (count + offset > m_AllocCount)
		{
			Log::Error("Downloaded too much from DeviceBuffer<>", true);
		}
		ASSERT_CUDA(cudaMemcpy(data, m_Ptr + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	constexpr void Download(T* data) const
	{
		DownloadSub(data, 0, m_Count);
	}

	constexpr CUdeviceptr GetCuPtr() const
	{
		return reinterpret_cast<CUdeviceptr>(m_Ptr);
	}

	constexpr size_t GetCount() const
	{
		return m_Count;
	}

	constexpr size_t GetByteSize() const
	{
		return m_Count * sizeof(T);
	}

private:
	size_t m_Count = 0;
	size_t m_AllocCount = 0;
	T* m_Ptr = nullptr;
};
