#include <graph/Scene.h>

Scene::Scene(const std::vector<ModelInstance>& modelInstances, const Pipeline& pipeline, const ShaderBindingTable& sbt) :
	m_ModelInstances(modelInstances),
	m_Pipeline(pipeline),
	m_Sbt(sbt)
{

}

Scene::~Scene()
{

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
