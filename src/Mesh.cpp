#include <model/Mesh.h>
#include <optix.h>

Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const Material* material) :
	m_Material(material)
{
	m_DeviceVertexBuffer.Alloc(vertices.size());
	m_DeviceVertexBuffer.Upload(vertices.data());

	m_DeviceIndexBuffer.Alloc(indices.size());
	m_DeviceIndexBuffer.Upload(indices.data());
}

OptixBuildInput Mesh::GetBuildInput(const OptixDeviceContext optixDeviceContext) const
{
	OptixBuildInput triangleInput{};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
	triangleInput.triangleArray.numVertices = m_DeviceVertexBuffer.GetCount();
	triangleInput.triangleArray.vertexBuffers = &m_VertexBufferDevicePtr;
	
	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(uint32_t);
	triangleInput.triangleArray.numIndexTriplets = m_DeviceIndexBuffer.GetCount();
	triangleInput.triangleArray.indexBuffer = m_DeviceIndexBuffer.GetCuPtr();

	triangleInput.triangleArray.flags = TRIANGLE_INPUT_FLAGS;
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	return triangleInput;
}
