#pragma once

#include <cstdint>
#include <vector>
#include <glad/glad.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <custom_assert.h>

template <typename T>
class OutputBuffer
{
public:
	constexpr OutputBuffer() = default;

	constexpr OutputBuffer(const uint32_t width, const uint32_t height) :
		m_Width(width),
		m_Height(height)
	{
		Resize(m_Width, m_Height);
	}

	constexpr void Resize(const uint32_t width, const uint32_t height)
	{
		if (width == m_Width && height == m_Height)
		{
			return;
		}
		
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
