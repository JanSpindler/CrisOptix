#pragma once

#include <glad/glad.h>
#include <Log.h>
#include <cuda.h>
#include <string>
#include <sstream>
#include <optix.h>

static std::string GetGlErrorStr(const GLenum result)
{
    switch (result)
    {
    case GL_NO_ERROR:            return "No error";
    case GL_INVALID_ENUM:        return "Invalid enum";
    case GL_INVALID_VALUE:       return "Invalid value";
    case GL_INVALID_OPERATION:   return "Invalid operation";
    //case GL_STACK_OVERFLOW:      return "Stack overflow";
    //case GL_STACK_UNDERFLOW:     return "Stack underflow";
    case GL_OUT_OF_MEMORY:       return "Out of memory";
    //case GL_TABLE_TOO_LARGE:     return "Table too large";
    default:                     return "Unknown GL error";
    }
}

#define CHECK_GL_ERROR() \
	{ \
        const GLenum result = glGetError(); \
		if (result != GL_NO_ERROR) \
		{ \
			std::stringstream ss; \
			ss << GetGlErrorStr(result); \
            ss << " at " << __FILE__ << ":" << __LINE__; \
            Log::Error(ss.str()); \
		} \
	}

#define ASSERT_CUDA(cudaResult) \
	{ \
		if (cudaResult != cudaSuccess) \
		{ \
			std::stringstream ss; \
			ss << cudaGetErrorString(cudaResult); \
            ss << " at " << __FILE__ << ":" << __LINE__; \
            Log::Error(ss.str()); \
		} \
	}

#define ASSERT_OPTIX(optixResult) \
	{ \
		if (optixResult != OPTIX_SUCCESS) \
		{ \
			std::stringstream ss; \
			ss << optixGetErrorName(optixResult); \
            ss << " at " << __FILE__ << ":" << __LINE__; \
            Log::Error(ss.str()); \
		} \
	}
