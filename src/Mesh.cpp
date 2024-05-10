#include <model/Mesh.h>

Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const Material* material)
{
	m_DeviceVertexBuffer.Alloc(vertices.size());
	m_DeviceVertexBuffer.Upload(vertices.data());

	m_DeviceIndexBuffer.Alloc(indices.size());
	m_DeviceIndexBuffer.Upload(indices.data());

	BuildAccelStructure();
}

void Mesh::BuildAccelStructure()
{

}
