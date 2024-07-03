#pragma once

#include <model/ModelInstance.h>

class Emitter
{
public:
	Emitter(const Mesh* mesh, const glm::mat4& transform, const glm::vec3& color);

private:
	glm::vec3 m_Color{};

	DeviceBuffer<Vertex> m_DeviceVertexBuffer{};
	DeviceBuffer<uint32_t> m_DeviceIndexBuffer{};
};
