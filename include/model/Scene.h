#pragma once

#include <optix.h>
#include <graph/Pipeline.h>
#include <graph/ShaderBindingTable.h>
#include <model/ModelInstance.h>
#include <model/Emitter.h>

class Scene
{
public:
	Scene(
		const OptixDeviceContext optixDeviceContext, 
		const std::vector<const ModelInstance*>& modelInstances,
		const std::vector<const Emitter*>& emitters);

	void AddShader(Pipeline& pipeline, ShaderBindingTable& sbt) const;

	OptixTraversableHandle GetTraversableHandle() const;
	CuBufferView<EmitterData> GetEmitterTable() const;

private:
	OptixTraversableHandle m_TraversableHandle = 0;
	const std::vector<const ModelInstance*>& m_ModelInstances;
	
	DeviceBuffer<uint8_t> m_AccelBuf{};
	DeviceBuffer<EmitterData> m_EmitterTable{};

	void BuildAccel(const OptixDeviceContext optixDeviceContext);
	void BuildEmitterTable(const std::vector<const Emitter*>& emitters);
};
