#pragma once

#include <optix.h>
#include <graph/Pipeline.h>
#include <graph/ShaderBindingTable.h>
#include <model/ModelInstance.h>

class Scene
{
public:
	Scene(const std::vector<ModelInstance>& modelInstances, const Pipeline& pipeline, const ShaderBindingTable& sbt);
	~Scene();

	OptixTraversableHandle GetTraversableHandle() const;
	const Pipeline& GetPipeline() const;
	const ShaderBindingTable& GetSbt() const;

private:
	OptixTraversableHandle m_TraversableHandle = 0;
	const std::vector<ModelInstance>& m_ModelInstances;
	const Pipeline& m_Pipeline;
	const ShaderBindingTable& m_Sbt;
};
