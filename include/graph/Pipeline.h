#pragma once

#include <optix.h>
#include <string>
#include <vector>
#include <unordered_map> 

struct ShaderEntryPointDesc
{
	OptixProgramGroupKind shaderKind;

	std::string fileName;
	std::string closestHitFileName;

	std::string entryPointName;
	std::string closestHitEntryPointName;

	std::string anyHitFileName;
	std::string anyHitEntryPointName;

	ShaderEntryPointDesc();
	~ShaderEntryPointDesc();

	ShaderEntryPointDesc(const ShaderEntryPointDesc& other);
};

class Pipeline
{
public:
	Pipeline(
		const OptixDeviceContext optixDeviceContext,
		const std::vector<ShaderEntryPointDesc>& shaders);
	~Pipeline();

	OptixPipeline GetHandle() const;

	const std::vector<OptixProgramGroup>& GetRaygenProgramGroups() const;
	const std::vector<OptixProgramGroup>& GetMissProgramGroups() const;
	const std::vector<OptixProgramGroup>& GetExceptionProgramGroups() const;
	const std::vector<OptixProgramGroup>& GetCallableProgramGroups() const;
	const std::vector<OptixProgramGroup>& GetHitgroupProgramGroups() const;

private:
	static inline const std::string SHADER_DIR = "CrisOptixShader.dir/Debug/";

	OptixPipeline m_Handle = nullptr;

	std::unordered_map<std::string, OptixModule> m_Modules{};

	std::vector<OptixProgramGroup> m_RaygenProgramGroups{};
	std::vector<OptixProgramGroup> m_MissProgramGroups{};
	std::vector<OptixProgramGroup> m_ExceptionProgramGroups{};
	std::vector<OptixProgramGroup> m_CallableProgramGroups{};
	std::vector<OptixProgramGroup> m_HitgroupProgramGroups{};

	OptixModule GetModule(
		const OptixDeviceContext optixDeviceContext,
		const std::string& fileName,
		const OptixModuleCompileOptions* moduleCompileOptions,
		const OptixPipelineCompileOptions* pipelineCompileOptions);
};
