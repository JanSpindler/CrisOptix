#include <graph/Scene.h>

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
