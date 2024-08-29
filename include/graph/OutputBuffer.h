#pragma once

#include <cstdint>
#include <vector>
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <util/custom_assert.h>

template <typename T>
class OutputBuffer
{
public:
	constexpr OutputBuffer(const uint32_t width, const uint32_t height)
	{
		Resize(width, height);
	}

	constexpr void Resize(const uint32_t width, const uint32_t height)
	{	
		m_Width = width;
		m_Height = height;

		if (m_Pbo == 0)
		{
			glGenBuffers(1, &m_Pbo);
		}

		const size_t bufferSize = m_Width * m_Height * sizeof(T);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		CHECK_GL_ERROR();

		ASSERT_CUDA(cudaGraphicsGLRegisterBuffer(&m_CudaGraphRes, m_Pbo, cudaGraphicsMapFlagsWriteDiscard));
	}

	constexpr void MapCuda()
	{
		size_t bufferSize = 0;
		ASSERT_CUDA(cudaGraphicsMapResources(1, &m_CudaGraphRes, m_Stream));
		ASSERT_CUDA(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&m_DevicePixels), &bufferSize, m_CudaGraphRes));
	}

	constexpr void UnmapCuda()
	{
		ASSERT_CUDA(cudaGraphicsUnmapResources(1, &m_CudaGraphRes, m_Stream));
	}

	constexpr CUdeviceptr GetPixelDevicePtr()
	{
		return reinterpret_cast<CUdeviceptr>(m_DevicePixels);
	}

	constexpr GLuint GetPbo() const
	{
		return m_Pbo;
	}

	constexpr std::vector<T> GetHostData()
	{
		// Assume: Already mapped

		std::vector<T> hostData(m_Width * m_Height);
		ASSERT_CUDA(cudaMemcpy(hostData.data(), m_DevicePixels, m_Width * m_Height * sizeof(T), cudaMemcpyDeviceToHost));
		return hostData;
	}

private:
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;

	cudaGraphicsResource* m_CudaGraphRes = nullptr;
	GLuint m_Pbo = 0;
	T* m_DevicePixels = nullptr;
	T* m_HostPixels = nullptr;

	CUstream m_Stream = 0;
};
