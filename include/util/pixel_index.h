#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <graph/LaunchParams.h>

static __forceinline__ __device__ __host__ size_t GetPixelIdx(const uint32_t x, const uint32_t y, const uint32_t width)
{
	return y * width + x;
}

static __forceinline__ __device__ __host__ size_t GetPixelIdx(const glm::uvec2& pixelCoord, const LaunchParams& params)
{
	return params.width * pixelCoord.y + pixelCoord.x;
}

static __forceinline__ __device__ __host__ bool IsPixelValid(const glm::uvec2& pixelCoord, const LaunchParams& params)
{
	return pixelCoord.x < params.width && pixelCoord.y < params.height;
}
