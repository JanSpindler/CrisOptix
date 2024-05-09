#pragma once

#include <cstdint>
#include <vector>
#include <glad/glad.h>
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

		glBindBuffer(GL_ARRAY_BUFFER, m_Pbo);
		glBufferData(GL_ARRAY_BUFFER, m_Width * m_Height * sizeof(T), nullptr, GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		CHECK_GL_ERROR();

		ASSERT_CUDA(cudaGraphicsGLRegisterBuffer(&m_CudaGraphRes, m_Pbo, cudaGraphicsMapFlagsWriteDiscard));

		m_PixelStorage.resize(m_Width * m_Height);
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

private:
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;

	cudaGraphicsResource* m_CudaGraphRes = nullptr;
	GLuint m_Pbo = 0;
	T* m_DevicePixels = nullptr;
	T* m_HostPixels = nullptr;
	std::vector<T> m_PixelStorage{};

	CUstream m_Stream = 0;
};
