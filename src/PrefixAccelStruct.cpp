#include <graph/restir/PrefixAccelStruct.h>
#include <array>

PrefixAccelStruct::PrefixAccelStruct(const size_t pixelCount, const size_t neighCount, const OptixDeviceContext context) :
	m_Context(context),
	m_Aabbs(pixelCount),
	m_AabbDevPtr(m_Aabbs.GetCuPtr()),
	m_PrefixNeighborBuf(pixelCount * neighCount),
	m_PixelCount(pixelCount),
	m_NeighCount(neighCount)
{
}

void PrefixAccelStruct::Rebuild(const size_t neighCount)
{
	// Resize neighbor pixel buffer
	if (neighCount != m_NeighCount)
	{
		m_NeighCount = neighCount;
		m_PrefixNeighborBuf.AllocIfRequired(m_NeighCount * m_PixelCount);
	}

	// Rebuild AS
	BuildGas();
	BuildTlas();
}

CuBufferView<OptixAabb> PrefixAccelStruct::GetAabbBufferView() const
{
	return CuBufferView<OptixAabb>(m_Aabbs.GetCuPtr(), m_Aabbs.GetCount());
}

CuBufferView<PrefixNeighbor> PrefixAccelStruct::GetPrefixNeighborBufferView() const
{
	return CuBufferView<PrefixNeighbor>(m_PrefixNeighborBuf.GetCuPtr(), m_PrefixNeighborBuf.GetCount());
}

CuBufferView<PrefixAccelStruct::Stats> PrefixAccelStruct::GetStatsBufferView() const
{
	return CuBufferView<PrefixAccelStruct::Stats>(m_StatsBuf.GetCuPtr(), m_StatsBuf.GetCount());
}

OptixTraversableHandle PrefixAccelStruct::GetTlas() const
{
	return m_TlasHandle;
}

PrefixAccelStruct::Stats PrefixAccelStruct::GetStats() const
{
	PrefixAccelStruct::Stats stats{};
	m_StatsBuf.Download(&stats);
	return stats;
}

void PrefixAccelStruct::SetSbtOffset(const uint32_t sbtOffset)
{
	m_SbtOffset = sbtOffset;
}

void PrefixAccelStruct::ResetStats()
{
	PrefixAccelStruct::Stats stats{};
	m_StatsBuf.Upload(&stats);
}

void PrefixAccelStruct::BuildGas()
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
		&m_GasHandle,
		&emitDesc,
		1));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());

	// Compact
	uint64_t compactedSize = 0;
	compactSizeBuf.Download(&compactedSize);

	m_GasBuf.Alloc(compactedSize);
	ASSERT_OPTIX(optixAccelCompact(
		m_Context,
		0,
		m_GasHandle,
		m_GasBuf.GetCuPtr(),
		m_GasBuf.GetByteSize(),
		&m_GasHandle));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());
}

void PrefixAccelStruct::BuildTlas()
{
	// Instance build
	static constexpr std::array<uint32_t, 1> flags = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

	OptixInstance optixInstance{};
	optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
	optixInstance.instanceId = 0;
	optixInstance.sbtOffset = m_SbtOffset;
	optixInstance.visibilityMask = 1;
	optixInstance.traversableHandle = m_GasHandle;
	reinterpret_cast<glm::mat3x4&>(optixInstance.transform) = glm::transpose(glm::mat4x3(glm::mat4(1.0f)));

	DeviceBuffer<OptixInstance> optixInstanceDev(1);
	optixInstanceDev.Upload(&optixInstance);

	OptixBuildInput instanceInput{};
	instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceInput.instanceArray.instances = optixInstanceDev.GetCuPtr();
	instanceInput.instanceArray.instanceStride = sizeof(OptixInstance);
	instanceInput.instanceArray.numInstances = optixInstanceDev.GetCount();

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
		&instanceInput,
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
		&instanceInput,
		1,
		tempBuf.GetCuPtr(),
		tempBuf.GetByteSize(),
		outputBuf.GetCuPtr(),
		outputBuf.GetByteSize(),
		&m_TlasHandle,
		&emitDesc,
		1));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());

	// Compact
	uint64_t compactedSize = 0;
	compactSizeBuf.Download(&compactedSize);

	m_TlasBuf.Alloc(compactedSize);
	ASSERT_OPTIX(optixAccelCompact(
		m_Context,
		0,
		m_TlasHandle,
		m_TlasBuf.GetCuPtr(),
		m_TlasBuf.GetByteSize(),
		&m_TlasHandle));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());
}
