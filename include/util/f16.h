#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <util/glm_cuda.h>

struct F16Vec2
{
	half x;
	half y;

	constexpr __forceinline__ __device__ __host__ F16Vec2(const half _x, const half _y) :
		x(_x),
		y(_y)
	{
	}

	__forceinline__ __device__ __host__ F16Vec2(const glm::vec2& vec) :
		x(vec.x),
		y(vec.y)
	{
	}

	__forceinline__ __device__ __host__ F16Vec2(const float2& vec) :
		x(vec.x),
		y(vec.y)
	{
	}

	constexpr operator glm::vec2() const 
	{
		return glm::vec2(x, y);
	}
	
	constexpr operator float2() const
	{
		return { x, y };
	}
};

struct F16Vec3
{
	half x;
	half y;
	half z;

	constexpr __forceinline__ __device__ __host__ F16Vec3(const half _x, const half _y, const half _z) :
		x(_x),
		y(_y),
		z(_z)
	{
	}

	__forceinline__ __device__ __host__ F16Vec3(const glm::vec3& vec) :
		x(vec.x),
		y(vec.y),
		z(vec.z)
	{
	}

	__forceinline__ __device__ __host__ F16Vec3(const float3& vec) :
		x(vec.x),
		y(vec.y),
		z(vec.z)
	{
	}

	constexpr __forceinline__ __device__ __host__ operator glm::vec3() const
	{
		return glm::vec3(x, y, z);
	}

	constexpr __forceinline__ __device__ __host__ operator float3() const
	{
		return { x, y, z };
	}
};
