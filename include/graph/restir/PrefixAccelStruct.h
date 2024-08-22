#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <graph/DeviceBuffer.h>
#include <graph/CuBufferView.h>

struct PrefixEntry
{
	glm::vec3 pos;
	uint32_t pixelIdx;

	__forceinline__ __device__ __host__ PrefixEntry() :
		pos(0.0f),
		pixelIdx(0)
	{
	}

	__forceinline__ __device__ __host__ PrefixEntry(const glm::vec3& _pos, const uint32_t _pixelIdx) :
		pos(_pos),
		pixelIdx(_pixelIdx)
	{
	}
};

struct PrefixEntryResult
{
	uint32_t neighCount;

	__forceinline__ __device__ __host__ PrefixEntryResult() :
		neighCount(0)
	{
	}
};

class PrefixAccelStruct
{
public:
	PrefixAccelStruct(const size_t size, const OptixDeviceContext context);

	void Rebuild();

	CuBufferView<OptixAabb> GetAabbBufferView() const;
	CuBufferView<PrefixEntry> GetPrefixEntryBufferView() const;
	OptixTraversableHandle GetTraversableHandle() const;

private:
	OptixDeviceContext m_Context = nullptr;

	OptixTraversableHandle m_TraversHandle = 0;
	DeviceBuffer<uint8_t> m_AccelBuf{};

	DeviceBuffer<OptixAabb> m_Aabbs{};
	CUdeviceptr m_AabbDevPtr = 0;

	DeviceBuffer<PrefixEntry> m_PrefixEntries{};

	void BuildAccel();
};
