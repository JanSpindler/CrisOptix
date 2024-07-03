#pragma once

#include <model/Vertex.h>
#include <vector>
#include <model/Material.h>
#include <graph/DeviceBuffer.h>
#include <array>
#include <graph/CuBufferView.h>

struct MeshSbtData
{
	CuBufferView<Vertex> vertices;
	CuBufferView<uint32_t> indices;
	uint32_t evalMaterialSbtIdx;
	//uint32_t sampleMaterialSbtIdx;
};

class Mesh
{
public:
	Mesh(
		const std::vector<Vertex>& vertices,
		const std::vector<uint32_t>& indices,
		const Material* material,
		const OptixDeviceContext optixDeviceContext);

	void AddShader(Pipeline& pipeline, ShaderBindingTable& sbt) const;

	OptixTraversableHandle GetTraversHandle() const;

private:
	static constexpr std::array<uint32_t, 1> m_TriangleInputFlags = { 0 };

	const Material* m_Material = nullptr;

	DeviceBuffer<Vertex> m_DeviceVertexBuffer{};
	DeviceBuffer<uint32_t> m_DeviceIndexBuffer{};
	CUdeviceptr m_VertexDevPtr = 0;

	mutable DeviceBuffer<MeshSbtData> m_SbtDataBuf{};

	OptixTraversableHandle m_TraversHandle = 0;
	DeviceBuffer<uint8_t> m_AccelBuf{};

	void BuildAccel(const OptixDeviceContext optixDeviceContext);
	void UploadSbtData() const;
};
