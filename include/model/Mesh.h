#pragma once

#include <model/Vertex.h>
#include <vector>
#include <model/Material.h>
#include <graph/DeviceBuffer.h>

class Mesh
{
public:
	Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const Material* material);

	OptixBuildInput GetBuildInput(const OptixDeviceContext optixDeviceContext) const;

private:
	const Material* m_Material = nullptr;

	DeviceBuffer<Vertex> m_DeviceVertexBuffer{};
	DeviceBuffer<uint32_t> m_DeviceIndexBuffer{};

	// Only needed for ptr to these variable
	CUdeviceptr m_VertexBufferDevicePtr = 0;
	static constexpr uint32_t TRIANGLE_INPUT_FLAGS = { 0 };
};
