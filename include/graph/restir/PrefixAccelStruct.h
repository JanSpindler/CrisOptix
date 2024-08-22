#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <graph/DeviceBuffer.h>
#include <graph/CuBufferView.h>

struct PrefixEntry
{
	bool valid;
	glm::vec3 pos;
	uint32_t pixelIdx;

	__forceinline__ __device__ __host__ PrefixEntry() :
		valid(false),
		pos(0.0f),
		pixelIdx(0)
	{
	}

	__forceinline__ __device__ __host__ PrefixEntry(const bool _valid, const glm::vec3& _pos, const uint32_t _pixelIdx) :
		valid(_valid),
		pos(_pos),
		pixelIdx(_pixelIdx)
	{
	}
};

struct PrefixEntryResult
{

};

class PrefixAccelStruct
{
public:
	PrefixAccelStruct(const size_t size, const OptixDeviceContext context);

	void Rebuild(const float radius);

	CuBufferView<PrefixEntry> GetPrefixEntryBufferView() const;
	OptixTraversableHandle GetTraversableHandle() const;

private:
	OptixDeviceContext m_Context = nullptr;

	OptixTraversableHandle m_TraversHandle = 0;
	DeviceBuffer<uint8_t> m_AccelBuf{};

	DeviceBuffer<PrefixEntry> m_PrefixEntries{};
	CUdeviceptr m_VertexDevPtr = 0;

	DeviceBuffer<float> m_RadiusBuffer = DeviceBuffer<float>(1);
	CUdeviceptr m_RadiusDevPtr = 0;

	void BuildAccel();
};
