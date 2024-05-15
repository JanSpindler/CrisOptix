#pragma once

#include <optix.h>
#include <string>
#include <vector>
#include <unordered_map> 

struct ShaderEntryPointDesc
{
	OptixProgramGroupKind shaderKind;

	union
	{
		std::string fileName;
		std::string closestHitFileName;
	};

	union
	{
		std::string entryPointName;
		std::string closestHitEntryPointName;
	};

	std::string anyHitFileName;
	std::string anyHitEntryPointName;
};

class Pipeline
{
public:
	Pipeline(
		const OptixDeviceContext optixDeviceContext,
		const std::vector<ShaderEntryPointDesc>& shaders);
	~Pipeline();

private:
	OptixPipeline m_Handle = nullptr;
	std::unordered_map<std::string, OptixModule> m_Modules{};
	std::vector<OptixProgramGroup> m_ProgramGroups{};

	OptixModule GetModule(
		const OptixDeviceContext optixDeviceContext,
		const std::string& fileName,
		const OptixModuleCompileOptions* moduleCompileOptions,
		const OptixPipelineCompileOptions* pipelineCompileOptions);
};
