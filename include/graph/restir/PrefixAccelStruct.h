#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <graph/DeviceBuffer.h>
#include <graph/CuBufferView.h>

struct PrefixNeighbor
{
	// Pixel index.
	uint32_t pixelIdx;

	// Distance to searching prefix.
	float distance;

	__forceinline__ __device__ __host__ PrefixNeighbor() :
		pixelIdx(0),
		distance(0.0f)
	{
	}

	__forceinline__ __device__ __host__ PrefixNeighbor(const uint32_t _pixelIdx, const float _distance) :
		pixelIdx(_pixelIdx),
		distance(_distance)
	{
	}
};

class PrefixAccelStruct
{
public:
	PrefixAccelStruct(const size_t size, const size_t neighCount, const OptixDeviceContext context);

	void Rebuild(const size_t neighCount);

	CuBufferView<OptixAabb> GetAabbBufferView() const;
	CuBufferView<PrefixNeighbor> GetPrefixNeighborBufferView() const;
	OptixTraversableHandle GetTlas() const;

	void SetSbtOffset(const uint32_t sbtOffset);

private:
	OptixDeviceContext m_Context = nullptr;

	OptixTraversableHandle m_GasHandle = 0;
	DeviceBuffer<uint8_t> m_GasBuf{};

	OptixTraversableHandle m_TlasHandle = 0;
	DeviceBuffer<uint8_t> m_TlasBuf{};

	DeviceBuffer<OptixAabb> m_Aabbs{};
	CUdeviceptr m_AabbDevPtr = 0;

	DeviceBuffer<PrefixNeighbor> m_PrefixNeighborBuf{};

	size_t m_PixelCount = 0;
	size_t m_NeighCount = 0;
	uint32_t m_SbtOffset = 0;

	void BuildGas();
	void BuildTlas();
};
