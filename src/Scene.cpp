#include <model/Scene.h>
#include <map>

Scene::Scene(
	const OptixDeviceContext optixDeviceContext, 
	const std::vector<ModelInstance>& modelInstances,
	const std::vector<Emitter>& emitter) 
	:
	m_ModelInstances(modelInstances),
	m_Emitter(emitter)
{
	const size_t modelInstanceCount = m_ModelInstances.size();
	std::vector<OptixInstance> optixInstances(modelInstanceCount);

	std::map<const Model*, size_t> modelSbtOffsetMap{};

	uint32_t sbtOffset = 0;
	for (size_t idx = 0; idx < modelInstanceCount; ++idx)
	{
		const ModelInstance& modelInstance = m_ModelInstances[idx];
		const Model& model = modelInstance.GetModel();

		// Use stored sbt offset if it exists
		size_t currentSbtOffset = sbtOffset;
		const bool modelNew = modelSbtOffsetMap.find(&model) == modelSbtOffsetMap.end();
		if (!modelNew) { currentSbtOffset = modelSbtOffsetMap[&model]; }

		// Construct OptixInstance
		OptixInstance& optixInstance = optixInstances[idx];
		optixInstance = {};
		optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		optixInstance.instanceId = idx;
		optixInstance.sbtOffset = sbtOffset;
		optixInstance.visibilityMask = 1;
		optixInstance.traversableHandle = modelInstance.GetModel().GetTraversHandle();
		reinterpret_cast<glm::mat3x4&>(optixInstance.transform) = glm::transpose(glm::mat4x3(modelInstance.GetTransform()));
		
		// Increase sbt offset if model was new
		if (modelNew) { sbtOffset += model.GetMeshCount(); }
	}

	DeviceBuffer<OptixInstance> optixDevInstances(modelInstanceCount);
	optixDevInstances.Upload(optixInstances.data());

	OptixBuildInput instanceInput{};
	instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceInput.instanceArray.instances = optixDevInstances.GetCuPtr();
	instanceInput.instanceArray.numInstances = modelInstanceCount;

	OptixAccelBuildOptions buildOptions{};
	buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.motionOptions.numKeys = 1;
	buildOptions.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
	buildOptions.motionOptions.timeBegin = 0.0f;
	buildOptions.motionOptions.timeEnd = 1.0f;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes accelBufferSizes{};
	ASSERT_OPTIX(optixAccelComputeMemoryUsage(optixDeviceContext, &buildOptions, &instanceInput, 1, &accelBufferSizes));

	DeviceBuffer<uint64_t> compactSizeBuf(1);
	DeviceBuffer<uint8_t> tempBuffer(accelBufferSizes.tempSizeInBytes);
	DeviceBuffer<uint8_t> outputBuffer(accelBufferSizes.outputSizeInBytes);

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactSizeBuf.GetCuPtr();

	// Build
	ASSERT_OPTIX(optixAccelBuild(
		optixDeviceContext,
		0,
		&buildOptions,
		&instanceInput,
		1,
		tempBuffer.GetCuPtr(),
		tempBuffer.GetByteSize(),
		outputBuffer.GetCuPtr(),
		outputBuffer.GetByteSize(),
		&m_TraversableHandle,
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
		m_TraversableHandle,
		m_AccelBuf.GetCuPtr(),
		m_AccelBuf.GetByteSize(),
		&m_TraversableHandle));

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());
}

void Scene::AddShader(Pipeline& pipeline, ShaderBindingTable& sbt) const
{
	// Add shader for each model in order
	for (size_t idx = 0; idx < m_ModelInstances.size(); ++idx)
	{
		m_ModelInstances[idx].GetModel().AddShader(pipeline, sbt);
	}
}

OptixTraversableHandle Scene::GetTraversableHandle() const
{
	return m_TraversableHandle;
}
