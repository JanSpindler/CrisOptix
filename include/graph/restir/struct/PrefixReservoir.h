#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct PrefixReservoir
{
	uint32_t McomponentType;
	float W;
	float reconJacobian;
	float pHat;
	float pathFootprint;

	constexpr __device__ PrefixReservoir() :
		McomponentType(1 << 2),
		W(1.0f),
		reconJacobian(0.0f),
		pHat(0.0f),
		pathFootprint(0.0f)
	{
	}

	constexpr __device__ PrefixReservoir(
		uint32_t _componentType, 
		bool needRandomReplay, 
		float _rcJacobian, 
		float _pHat, 
		float _pathFootprint, 
		float _W = 1.f, 
		int M = 1)
		:
		McomponentType(_componentType),
		W(_W),
		reconJacobian(_rcJacobian),
		pHat(_pHat),
		pathFootprint(_pathFootprint)
	{
		McomponentType = (M << 3) | (int(needRandomReplay) << 2) | _componentType;
	}

	constexpr __device__ int M() const
	{
		return McomponentType >> 3;
	}

	constexpr __device__ void setM(int M)
	{
		McomponentType &= 0x7;
		McomponentType |= M << 3;
	}

	constexpr __device__ void increaseM(int inc)
	{
		McomponentType += inc << 3;
	}

	constexpr __device__ void setNeedRandomReplay(bool val)
	{
		McomponentType &= ~0x4;
		McomponentType |= int(val) << 2;
	}

	constexpr __device__ bool needRandomReplay() const
	{
		return (McomponentType >> 2) & 1;
	}

	constexpr __device__ uint32_t componentType() const
	{
		return McomponentType & 0x3;
	}

	constexpr __device__ void setComponentType(uint32_t type)
	{
		McomponentType &= ~0x3;
		McomponentType |= type;
	}
};
