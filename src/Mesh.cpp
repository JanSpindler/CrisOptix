#include <model/Mesh.h>
#include <optix.h>

Mesh::Mesh(
	const std::vector<Vertex>& vertices, 
	const std::vector<uint32_t>& indices, 
	const Material* material,
	const OptixDeviceContext optixDeviceContext)
	:
	m_Material(material)
{
	// Vertex buffer
	m_DeviceVertexBuffer.Alloc(vertices.size());
	m_DeviceVertexBuffer.Upload(vertices.data());

	// Index buffer
	m_DeviceIndexBuffer.Alloc(indices.size());
	m_DeviceIndexBuffer.Upload(indices.data());

	// Helper variable
	m_VertexDevPtr = m_DeviceVertexBuffer.GetCuPtr();

	// Build accel
	BuildAccel(optixDeviceContext);
}

void Mesh::AddShader(Pipeline& pipeline, ShaderBindingTable& sbt) const
{
	UploadSbtData();
	const OptixProgramGroup pg = pipeline.AddTrianglesHitGroupShader({ "test.ptx", "__closesthit__mesh" }, {});
	sbt.AddHitEntry(pg, ToVecByte(m_SbtDataBuf.GetCuPtr()));
}

OptixTraversableHandle Mesh::GetTraversHandle() const
{
	return m_TraversHandle;
}

const DeviceBuffer<Vertex>& Mesh::GetDeviceVertexBuffer() const
{
	return m_DeviceVertexBuffer;
}

const DeviceBuffer<uint32_t>& Mesh::GetDeviceIndexBuffer() const
{
	return m_DeviceIndexBuffer;
}

const Material* Mesh::GetMaterial() const
{
	return m_Material;
}

void Mesh::BuildAccel(const OptixDeviceContext optixDeviceContext)
{
	// Triangle input
	OptixBuildInput triangleInput{};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
	triangleInput.triangleArray.numVertices = m_DeviceVertexBuffer.GetCount();
	triangleInput.triangleArray.vertexBuffers = &m_VertexDevPtr;

	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
	triangleInput.triangleArray.numIndexTriplets = m_DeviceIndexBuffer.GetCount() / 3;
	triangleInput.triangleArray.indexBuffer = m_DeviceIndexBuffer.GetCuPtr();

	triangleInput.triangleArray.flags = m_TriangleInputFlags.data();
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	// Accel
	// Get memory requirements
	OptixAccelBuildOptions buildOptions{};
	buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.motionOptions.numKeys = 1;
	buildOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
	buildOptions.motionOptions.timeBegin = 0.0f;
	buildOptions.motionOptions.timeEnd = 1.0f;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes{};
	ASSERT_OPTIX(optixAccelComputeMemoryUsage(
		optixDeviceContext,
		&buildOptions,
		&triangleInput,
		1,
		&blasBufferSizes));

	// Build
	DeviceBuffer<uint64_t> compactSizeBuf(1);
	DeviceBuffer<uint8_t> tempBuf(blasBufferSizes.tempSizeInBytes);
	DeviceBuffer<uint8_t> outputBuf(blasBufferSizes.outputSizeInBytes);

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactSizeBuf.GetCuPtr();

	ASSERT_OPTIX(optixAccelBuild(
		optixDeviceContext,
		0,
		&buildOptions,
		&triangleInput,
		1,
		tempBuf.GetCuPtr(),
		tempBuf.GetByteSize(),
		outputBuf.GetCuPtr(),
		outputBuf.GetByteSize(),
		&m_TraversHandle,
		&emitDesc,
		1));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());

	// Compact
	uint64_t compactedSize = 0;
	compactSizeBuf.Download(&compactedSize);

	m_AccelBuf.Alloc(compactedSize);
	ASSERT_OPTIX(optixAccelCompact(
		optixDeviceContext,
		0,
		m_TraversHandle,
		m_AccelBuf.GetCuPtr(),
		m_AccelBuf.GetByteSize(),
		&m_TraversHandle));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());
}

void Mesh::UploadSbtData() const
{
	MeshSbtData data{};
	data.vertices = CuBufferView<Vertex>(m_DeviceVertexBuffer.GetCuPtr(), m_DeviceVertexBuffer.GetCount());
	data.indices = CuBufferView<uint32_t>(m_DeviceIndexBuffer.GetCuPtr(), m_DeviceIndexBuffer.GetCount());
	data.evalMaterialSbtIdx = m_Material->GetEvalSbtIdx();
	data.sampleMaterialSbtIdx = m_Material->GetSampleSbtIdx();
	m_SbtDataBuf.Alloc(1);
	m_SbtDataBuf.Upload(&data);
}
