#include <graph/Pipeline.h>
#include <util/custom_assert.h>
#include <optix_stubs.h>
#include <array>
#include <util/read_file.h>

Pipeline::Pipeline(
	const OptixDeviceContext optixDeviceContext,
	const std::vector<ShaderEntryPointDesc>& shaders)
{
	OptixModuleCompileOptions moduleCompileOptions{};
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
	moduleCompileOptions.boundValues = nullptr;
	moduleCompileOptions.numBoundValues = 0;
	moduleCompileOptions.numPayloadTypes = 0;
	moduleCompileOptions.payloadTypes = nullptr;

	OptixPipelineCompileOptions pipelineCompileOptions{};
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	pipelineCompileOptions.numPayloadValues = 3;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

	for (const ShaderEntryPointDesc& shaderDesc : shaders)
	{
		OptixModule singleModule = nullptr;
		OptixModule closestHitModule = nullptr;
		OptixModule anyHitModule = nullptr;
		if (shaderDesc.shaderKind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP)
		{
			closestHitModule = GetModule(
				optixDeviceContext,
				shaderDesc.closestHitFileName,
				&moduleCompileOptions,
				&pipelineCompileOptions);
			anyHitModule = GetModule(
				optixDeviceContext,
				shaderDesc.anyHitFileName,
				&moduleCompileOptions,
				&pipelineCompileOptions);
		}
		else
		{
			singleModule = GetModule(
				optixDeviceContext,
				shaderDesc.fileName,
				&moduleCompileOptions,
				&pipelineCompileOptions);
		}

		OptixProgramGroupDesc programGroupDesc{};
		programGroupDesc.kind = shaderDesc.shaderKind;
		programGroupDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
		
		switch (shaderDesc.shaderKind)
		{
		case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
			programGroupDesc.raygen.module = singleModule;
			programGroupDesc.raygen.entryFunctionName = shaderDesc.entryPointName.c_str();
			break;
		case OPTIX_PROGRAM_GROUP_KIND_MISS:
			programGroupDesc.miss.module = singleModule;
			programGroupDesc.miss.entryFunctionName = shaderDesc.entryPointName.c_str();
			break;
		case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
			programGroupDesc.exception.module = singleModule;
			programGroupDesc.exception.entryFunctionName = shaderDesc.entryPointName.c_str();
			break;
		case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
			if (shaderDesc.entryPointName.rfind("__direct_callable__", 0) == 0)
			{
				programGroupDesc.callables.moduleDC = singleModule;
				programGroupDesc.callables.entryFunctionNameDC = shaderDesc.entryPointName.c_str();
			}
			else if (shaderDesc.entryPointName.rfind("__continuation_callable__", 0) == 0)
			{
				programGroupDesc.callables.moduleCC = singleModule;
				programGroupDesc.callables.entryFunctionNameCC = shaderDesc.entryPointName.c_str();
			}
			else
			{
				Log::Error("Callable program type invalid", true);
			}
			break;
		case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
			programGroupDesc.hitgroup.moduleCH = closestHitModule;
			programGroupDesc.hitgroup.entryFunctionNameCH = shaderDesc.closestHitEntryPointName.c_str();
			programGroupDesc.hitgroup.moduleAH = anyHitModule;
			programGroupDesc.hitgroup.entryFunctionNameAH = shaderDesc.anyHitEntryPointName.c_str();
			break;
		default:
			Log::Error("Invalid shader kind", true);
			break;
		}

		char log[2028];
		size_t logSize = 2028;

		OptixProgramGroupOptions programGroupOptions{};
		OptixProgramGroup programGroup = nullptr;
		ASSERT_OPTIX(optixProgramGroupCreate(
			optixDeviceContext, 
			&programGroupDesc, 
			1, 
			&programGroupOptions, 
			log, 
			&logSize, 
			&programGroup));

		Log::Assert(logSize == std::strlen(log) && logSize < 2028);
		Log::Info(log);

		m_ProgramGroups.push_back(programGroup);
	}

	OptixPipelineLinkOptions pipelineLinkOptions{};
	pipelineLinkOptions.maxTraceDepth = 8;
}

Pipeline::~Pipeline()
{
	ASSERT_OPTIX(optixPipelineDestroy(m_Handle));

	for (const auto& [key, optixModule] : m_Modules)
	{
		ASSERT_OPTIX(optixModuleDestroy(optixModule));
	}

	for (const OptixProgramGroup programGroup : m_ProgramGroups)
	{
		ASSERT_OPTIX(optixProgramGroupDestroy(programGroup));
	}
}

OptixModule Pipeline::GetModule(
	const OptixDeviceContext optixDeviceContext,
	const std::string& fileName,
	const OptixModuleCompileOptions* moduleCompileOptions,
	const OptixPipelineCompileOptions* pipelineCompileOptions)
{
	Log::Assert(!fileName.empty());

	if (m_Modules.find(fileName) != m_Modules.end())
	{
		return m_Modules[fileName];
	}

	std::vector<char> src = ReadFile(fileName);
	
	char log[2028];
	size_t logSize = 2028;

	OptixModule optixModule = nullptr;
	ASSERT_OPTIX(optixModuleCreate(
		optixDeviceContext, 
		moduleCompileOptions,
		pipelineCompileOptions, 
		src.data(), 
		src.size(), 
		log, 
		&logSize, 
		&optixModule));

	Log::Assert(std::strlen(log) < 2028 && std::strlen(log) == logSize);
	Log::Info(std::string(log));

	m_Modules[fileName] = optixModule;
	return optixModule;
}