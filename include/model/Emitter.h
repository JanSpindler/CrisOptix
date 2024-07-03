#pragma once

#include <model/ModelInstance.h>

struct EmitterData
{
	glm::vec3 color;
	float totalArea;
	CuBufferView<Vertex> vertexBuffer;
	CuBufferView<uint32_t> indexBuffer;
	CuBufferView<float> areaBuffer;
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
	DeviceBuffer<float> m_DeviceAreaBuffer{};
};
