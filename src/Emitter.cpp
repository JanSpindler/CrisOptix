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

	// Calculate areas
	m_TotalArea = 0.0f;
	std::vector<float> areas(indices.size() / 3);
	for (size_t faceIdx = 0; faceIdx < areas.size(); ++faceIdx)
	{
		const glm::vec3 v0 = vertices[indices[faceIdx * 3 + 0]].pos;
		const glm::vec3 v1 = vertices[indices[faceIdx * 3 + 1]].pos;
		const glm::vec3 v2 = vertices[indices[faceIdx * 3 + 2]].pos;

		const float a = glm::distance(v0, v1);
		const float b = glm::distance(v1, v2);
		const float c = glm::distance(v2, v0);

		const float s = (a + b + c) * 0.5f;
		const float area = glm::sqrt(s * (s - a) * (s - b) * (s - c));

		areas[faceIdx] = area;
		m_TotalArea += area;
	}

	// Upload areas
	m_DeviceAreaBuffer.Alloc(areas.size());
	m_DeviceAreaBuffer.Upload(areas.data());
}

EmitterData Emitter::GetData() const
{
	return {
		m_Color,
		m_TotalArea,
		CuBufferView<Vertex>(m_DeviceVertexBuffer.GetCuPtr(), m_DeviceVertexBuffer.GetCount()),
		CuBufferView<uint32_t>(m_DeviceIndexBuffer.GetCuPtr(), m_DeviceIndexBuffer.GetCount()),
		CuBufferView<float>(m_DeviceAreaBuffer.GetCuPtr(), m_DeviceAreaBuffer.GetCount())
	};
}
