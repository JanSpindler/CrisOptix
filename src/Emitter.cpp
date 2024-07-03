#include <model/Emitter.h>

Emitter::Emitter(const Mesh* mesh, const glm::mat4& transform, const glm::vec3& color) :
	m_Color(color)
{
	// Download index buffer
	std::vector<uint32_t> indices(mesh->GetDeviceIndexBuffer().GetCount());
	mesh->GetDeviceIndexBuffer().Download(indices.data());

	// Upload index buffer
	m_DeviceIndexBuffer.Alloc(indices.size());
	m_DeviceIndexBuffer.Upload(indices.data());

	// Download vertex buffer
	std::vector<Vertex> vertices(mesh->GetDeviceVertexBuffer().GetCount());
	mesh->GetDeviceVertexBuffer().Download(vertices.data());

	// Apply transform to vertices
	for (Vertex& vertex : vertices)
	{
		vertex.pos = glm::vec3(transform * glm::vec4(vertex.pos, 1.0f));
		vertex.normal = glm::mat3(transform) * vertex.normal;
		vertex.tangent = glm::mat3(transform) * vertex.tangent;
	}

	// Upload transformed vertices
	m_DeviceVertexBuffer.Alloc(vertices.size());
	m_DeviceVertexBuffer.Upload(vertices.data());
}
