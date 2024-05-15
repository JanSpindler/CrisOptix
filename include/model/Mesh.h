#pragma once

#include <model/Vertex.h>
#include <vector>
#include <model/Material.h>
#include <graph/DeviceBuffer.h>
#include <array>

class Mesh
{
public:
	Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const Material* material);

	OptixBuildInput GetBuildInput(const OptixDeviceContext optixDeviceContext) const;

private:
	const Material* m_Material = nullptr;

	DeviceBuffer<Vertex> m_DeviceVertexBuffer{};
	DeviceBuffer<uint32_t> m_DeviceIndexBuffer{};

	CUdeviceptr m_VertexDevPtr = 0;
	static constexpr std::array<uint32_t, 1> m_TriangleInputFlags = { 0 };
};
