#pragma once

#include <model/ModelInstance.h>
#include <util/random.h>

struct EmitterSample
{
	glm::vec3 pos;
	glm::vec3 color;
	float p;
};

struct EmitterData
{
	glm::vec3 color;
	float totalArea;
	CuBufferView<Vertex> vertexBuffer;
	CuBufferView<uint32_t> indexBuffer;
	CuBufferView<float> areaCdfBuffer;

	__host__ __device__ size_t SampleFaceAreaWeighted(PCG32& rng) const
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

	__host__ __device__ EmitterSample SamplePoint(PCG32& rng) const
	{
		// Get face
		const size_t faceIdx = SampleFaceAreaWeighted(rng);
		
		// Sample emitter point on face
		const glm::vec3 v0 = vertexBuffer[indexBuffer[faceIdx * 3 + 0]].pos;
		const glm::vec3 v1 = vertexBuffer[indexBuffer[faceIdx * 3 + 1]].pos;
		const glm::vec3 v2 = vertexBuffer[indexBuffer[faceIdx * 3 + 2]].pos;

		float r0 = rng.NextFloat();
		float r1 = rng.NextFloat();
		float r2 = rng.NextFloat();
		const float rSum = r0 + r1 + r2;
		r0 /= rSum;
		r1 /= rSum;
		r2 /= rSum;

		const glm::vec3 emitterPoint = r0 * v0 + r1 * v1 + r2 * v2;
		return { emitterPoint, color, 1.0f };
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
