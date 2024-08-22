#include <graph/restir/PrefixAccelStruct.h>
#include <array>

PrefixAccelStruct::PrefixAccelStruct(const size_t size, const OptixDeviceContext context) :
	m_Context(context),
	m_Aabbs(size),
	m_AabbDevPtr(m_Aabbs.GetCuPtr()),
	m_PrefixEntries(size)
{
}

void PrefixAccelStruct::Rebuild()
{
	// Build accel
	BuildAccel();
}

CuBufferView<OptixAabb> PrefixAccelStruct::GetAabbBufferView() const
{
	return CuBufferView<OptixAabb>(m_Aabbs.GetCuPtr(), m_Aabbs.GetCount());
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
	// AABB build
	static constexpr std::array<uint32_t, 1> flags = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

	OptixBuildInput aabbInput{};
	aabbInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	aabbInput.customPrimitiveArray.aabbBuffers = &m_AabbDevPtr;
	aabbInput.customPrimitiveArray.numPrimitives = m_Aabbs.GetCount();
	aabbInput.customPrimitiveArray.flags = flags.data();
	aabbInput.customPrimitiveArray.numSbtRecords = 1;
	aabbInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;

	// Accel
	// Get memory requirements
	OptixAccelBuildOptions buildOptions{};
	buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	buildOptions.motionOptions.numKeys = 1;
	buildOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
	buildOptions.motionOptions.timeBegin = 0.0f;
	buildOptions.motionOptions.timeEnd = 1.0f;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes{};
	ASSERT_OPTIX(optixAccelComputeMemoryUsage(
		m_Context,
		&buildOptions,
		&aabbInput,
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
		&aabbInput,
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
