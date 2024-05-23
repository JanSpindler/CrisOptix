#pragma once

#include <optix.h>
#include <graph/Pipeline.h>
#include <graph/ShaderBindingTable.h>

class Scene
{
public:
	OptixTraversableHandle GetTraversableHandle() const;
	const Pipeline& GetPipeline() const;
	const ShaderBindingTable& GetSbt() const;

private:
	OptixTraversableHandle m_TraversableHandle = 0;
	const Pipeline& m_Pipeline;
	const ShaderBindingTable& m_Sbt;
};
