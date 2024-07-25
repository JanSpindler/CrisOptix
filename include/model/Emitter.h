#pragma once

#include <model/ModelInstance.h>
#include <util/random.h>

struct EmitterData
{
	glm::vec3 color;
	float totalArea;
	CuBufferView<Vertex> vertexBuffer;
	CuBufferView<uint32_t> indexBuffer;
	CuBufferView<float> areaCdfBuffer;

	__device__ __host__ size_t SampleFaceAreaWeighted(PCG32& rng) const
	{
		//const float randCumArea = rng.NextFloat() * totalArea;
		//for (size_t faceIdx = 0; faceIdx < areaCdfBuffer.count; ++faceIdx)
		//{
		//	if (areaCdfBuffer[faceIdx] > randCumArea)
		//	{
		//		return faceIdx;
		//	}
		//}
		return rng.NextUint64() % areaCdfBuffer.count;
	}
};

class Emitter
{
public:
	Emitter(const Mesh* mesh, const glm::mat4& transform, const glm::vec3& color);

	EmitterData GetData() const;

private:
	glm::vec3 m_Color{};
	float m_TotalArea = 0.0f;

	DeviceBuffer<Vertex> m_DeviceVertexBuffer{};
	DeviceBuffer<uint32_t> m_DeviceIndexBuffer{};
	DeviceBuffer<float> m_DeviceAreaCdfBuffer{};
};
