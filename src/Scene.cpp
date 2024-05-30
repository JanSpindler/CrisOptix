#include <graph/Scene.h>

Scene::Scene(const OptixDeviceContext optixDeviceContext,
	const std::vector<ModelInstance>& modelInstances,
	const Pipeline& pipeline,
	const ShaderBindingTable& sbt) 
	:
	m_ModelInstances(modelInstances),
	m_Pipeline(pipeline),
	m_Sbt(sbt)
{
	static constexpr uint32_t rayCount = 8; // TODO

	const size_t modelInstanceCount = m_ModelInstances.size();
	std::vector<OptixInstance> optixInstances(modelInstanceCount);

	uint32_t sbtOffset = 0;
	for (size_t idx = 0; idx < modelInstanceCount; ++idx)
	{
		const ModelInstance& modelInstance = m_ModelInstances[idx];
		OptixInstance& optixInstance = optixInstances[idx];
		optixInstance = {};
		optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		optixInstance.instanceId = idx;
		optixInstance.sbtOffset = sbtOffset;
		optixInstance.visibilityMask = 1;
		optixInstance.traversableHandle = modelInstance.GetModel().GetTraversableHandle();
		reinterpret_cast<glm::mat3x4&>(optixInstance.transform) = glm::transpose(glm::mat4x3(modelInstance.GetModelMat()));
		sbtOffset += rayCount;
	}

	DeviceBuffer<OptixInstance> optixDevInstances(modelInstanceCount);
	optixDevInstances.Upload(optixInstances.data());

	OptixBuildInput instanceInput{};
	instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceInput.instanceArray.instances = optixDevInstances.GetCuPtr();
	instanceInput.instanceArray.numInstances = modelInstanceCount;

	OptixAccelBuildOptions buildOptions{};
	buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes accelBufferSizes{};
	ASSERT_OPTIX(optixAccelComputeMemoryUsage(optixDeviceContext, &buildOptions, &instanceInput, 1, &accelBufferSizes));

	DeviceBuffer<uint8_t> tempBuffer(accelBufferSizes.tempSizeInBytes);
	m_AccelBuf.Alloc(accelBufferSizes.outputSizeInBytes);

	ASSERT_OPTIX(optixAccelBuild(
		optixDeviceContext,
		0,
		&buildOptions,
		&instanceInput,
		1,
		tempBuffer.GetCuPtr(),
		tempBuffer.GetByteSize(),
		m_AccelBuf.GetCuPtr(),
		m_AccelBuf.GetByteSize(),
		&m_TraversableHandle,
		nullptr,
		0));
}

OptixTraversableHandle Scene::GetTraversableHandle() const
{
	return m_TraversableHandle;
}

const Pipeline& Scene::GetPipeline() const
{
	return m_Pipeline;
}

const ShaderBindingTable& Scene::GetSbt() const
{
	return m_Sbt;
}

void Scene::BuildTlas()
{

}
