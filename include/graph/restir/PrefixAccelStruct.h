#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <graph/DeviceBuffer.h>
#include <graph/CuBufferView.h>

struct PrefixEntry
{
	glm::vec3 pos;
	uint32_t pixelIdx;
};

class PrefixAccelStruct
{
public:
	PrefixAccelStruct(const size_t size, const OptixDeviceContext context);

	void Rebuild(const float radius);

	CuBufferView<PrefixEntry> GetPrefixEntryBuffer() const;

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
