#pragma once

#include <model/Vertex.h>
#include <vector>
#include <model/Material.h>
#include <cuda_runtime.h>
#include <graph/DeviceBuffer.h>

class Mesh
{
public:
	Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const Material* material);
	
private:
	DeviceBuffer<Vertex> m_DeviceVertexBuffer{};
	DeviceBuffer<uint32_t> m_DeviceIndexBuffer{};

	void BuildAccelStructure();
};
