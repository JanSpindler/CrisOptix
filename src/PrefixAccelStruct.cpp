#include <graph/restir/PrefixAccelStruct.h>
#include <array>

PrefixAccelStruct::PrefixAccelStruct(const size_t size, const OptixDeviceContext context) :
	m_Context(context),
	m_PrefixEntries(size),
	m_VertexDevPtr(m_PrefixEntries.GetCuPtr()),
	m_RadiusDevPtr(m_RadiusBuffer.GetCuPtr())
{
}

void PrefixAccelStruct::Rebuild(const float radius)
{
	// Upload radius to buffer
	m_RadiusBuffer.Upload(&radius);

	// Build accel
	BuildAccel();
}

CuBufferView<PrefixEntry> PrefixAccelStruct::GetPrefixEntryBufferView() const
{
	return CuBufferView<PrefixEntry>(m_PrefixEntries.GetCuPtr(), m_PrefixEntries.GetCount());
}

OptixTraversableHandle PrefixAccelStruct::GetTraversableHandle() const
{
	return m_TraversHandle;
}

void PrefixAccelStruct::BuildAccel()
{
	// Sphere build
	OptixBuildInput sphereInput{};
	sphereInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

	sphereInput.sphereArray.vertexBuffers = &m_VertexDevPtr;
	sphereInput.sphereArray.vertexStrideInBytes = sizeof(PrefixEntry);
	sphereInput.sphereArray.numVertices = m_PrefixEntries.GetCount();

	sphereInput.sphereArray.radiusBuffers = &m_RadiusDevPtr;
	sphereInput.sphereArray.radiusStrideInBytes = 0;
	sphereInput.sphereArray.singleRadius = true;
	
	static constexpr std::array<uint32_t, 1> flags = { 0 };
	sphereInput.sphereArray.flags = flags.data();
	
	sphereInput.sphereArray.numSbtRecords = 1;
	sphereInput.sphereArray.sbtIndexOffsetBuffer = 0;
	sphereInput.sphereArray.sbtIndexOffsetSizeInBytes = 0;
	sphereInput.sphereArray.sbtIndexOffsetStrideInBytes = 0;

	sphereInput.sphereArray.primitiveIndexOffset = 0;

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
		m_Context,
		&buildOptions,
		&sphereInput,
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
		m_Context,
		0,
		&buildOptions,
		&sphereInput,
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
		m_Context,
		0,
		m_TraversHandle,
		m_AccelBuf.GetCuPtr(),
		m_AccelBuf.GetByteSize(),
		&m_TraversHandle));

	// Sync
	// TODO: Is this sync needed?
	ASSERT_CUDA(cudaDeviceSynchronize());
}
