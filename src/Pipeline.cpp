#include <algorithm>
#include <optix_stubs.h>
#include <graph/Pipeline.h>
#include <util/custom_assert.h>
#include <util/read_file.h>
#include <graph/Interaction.h>

Pipeline::Pipeline(OptixDeviceContext context) :
    m_Context { context }
{
    // Set default module compile options
    m_ModuleCompileOptions.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    m_ModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;//OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_ModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;//OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    // Set default pipeline compile options
    m_PipelineCompileOptions.usesMotionBlur        = false;
    m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; // OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_PipelineCompileOptions.numPayloadValues      = 3; // radiance uses 3, occlusion uses 1
    m_PipelineCompileOptions.numAttributeValues    = 2; // two values for barycentric coordinates on a triangle
    m_PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    // Set default pipeline link options
    m_PipelineLinkOptions.maxTraceDepth            = 2;//MAX_TRACE_DEPTH;
}

Pipeline::~Pipeline()
{
    optixPipelineDestroy(m_Handle);

    for (const auto & [ key, prog_group ] : m_ProgramGroups)
    {
        optixProgramGroupDestroy(prog_group);
    }

    for (const auto & [ filename, module ] : m_ModuleCache)
    {
        optixModuleDestroy(module);
    }
}

OptixProgramGroup Pipeline::CreateNewProgramGroup(const OptixProgramGroupDesc &prog_group_desc)
{
    // TODO this needs to be cached?!?!
    // Yes, this is indeed the correct level for caching..
    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);

    OptixProgramGroup prog_group;
    ASSERT_OPTIX( optixProgramGroupCreate( m_Context, &prog_group_desc,
                                            1,  // num program groups
                                            &m_ProgramGroupOptions, log, &sizeof_log, &prog_group ) );

    return prog_group;
}

OptixProgramGroup Pipeline::GetCachedProgramGroup(const OptixProgramGroupDesc &prog_group_desc)
{
    OptixProgramGroup &prog_group = m_ProgramGroups[prog_group_desc];
    if (prog_group == nullptr)
        prog_group = CreateNewProgramGroup(prog_group_desc);
    return prog_group;
}

OptixProgramGroup Pipeline::AddRaygenShader(const ShaderEntryPointDesc &raygen_shader_desc)
{
    OptixModule ptx_module = GetCachedModule(raygen_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc    = {};
    prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    prog_group_desc.raygen.module            = ptx_module;
    prog_group_desc.raygen.entryFunctionName = raygen_shader_desc.entrypoint_name.c_str();

    return GetCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup Pipeline::AddCallableShader(const ShaderEntryPointDesc &callable_shader_desc)
{
    OptixModule ptx_module = GetCachedModule(callable_shader_desc.ptx_filename);

    // TODO directs VS continuation callable!!!

    OptixProgramGroupDesc prog_group_desc         = {};
    prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

    if (callable_shader_desc.entrypoint_name.rfind("__direct_callable__", 0) == 0)
    {
        prog_group_desc.callables.moduleDC            = ptx_module;
        prog_group_desc.callables.entryFunctionNameDC = callable_shader_desc.entrypoint_name.c_str();
    }
    else if (callable_shader_desc.entrypoint_name.rfind("__continuation_callable__", 0) == 0)
    {
        prog_group_desc.callables.moduleCC            = ptx_module;
        prog_group_desc.callables.entryFunctionNameCC = callable_shader_desc.entrypoint_name.c_str();
    }
    else
    {
        std::stringstream ss;
        ss << "Not a callable program entry point: " << callable_shader_desc.entrypoint_name;
        throw std::runtime_error(ss.str());
    }

    return GetCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup Pipeline::AddMissShader(const ShaderEntryPointDesc &miss_shader_desc)
{
    OptixModule ptx_module = GetCachedModule(miss_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc  = {};
    prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    prog_group_desc.miss.module            = ptx_module;
    prog_group_desc.miss.entryFunctionName = miss_shader_desc.entrypoint_name.c_str();

    return GetCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup Pipeline::AddTrianglesHitGroupShader(const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc)
{
    OptixModule ptx_module_ch = GetCachedModule(closestHit_shader_desc.ptx_filename);
    OptixModule ptx_module_ah = GetCachedModule(anyHit_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc        = {};
    prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc.hitgroup.moduleCH            = ptx_module_ch;
    prog_group_desc.hitgroup.entryFunctionNameCH = ptx_module_ch ? closestHit_shader_desc.entrypoint_name.c_str() : nullptr;
    prog_group_desc.hitgroup.moduleAH            = ptx_module_ah;
    prog_group_desc.hitgroup.entryFunctionNameAH = ptx_module_ah ? anyHit_shader_desc.entrypoint_name.c_str() : nullptr;

    return GetCachedProgramGroup(prog_group_desc);
}

OptixProgramGroup Pipeline::AddProceduralHitGroupShader(const ShaderEntryPointDesc &intersection_shader_desc, const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc)
{
    OptixModule ptx_module_ch = GetCachedModule(closestHit_shader_desc.ptx_filename);
    OptixModule ptx_module_ah = GetCachedModule(anyHit_shader_desc.ptx_filename);
    OptixModule ptx_module_is = GetCachedModule(intersection_shader_desc.ptx_filename);

    OptixProgramGroupDesc prog_group_desc        = {};
    prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc.hitgroup.moduleCH            = ptx_module_ch;
    prog_group_desc.hitgroup.entryFunctionNameCH = ptx_module_ch ? closestHit_shader_desc.entrypoint_name.c_str() : nullptr;
    prog_group_desc.hitgroup.moduleAH            = ptx_module_ah;
    prog_group_desc.hitgroup.entryFunctionNameAH = ptx_module_ah ? anyHit_shader_desc.entrypoint_name.c_str() : nullptr;
    prog_group_desc.hitgroup.moduleIS            = ptx_module_is;
    prog_group_desc.hitgroup.entryFunctionNameIS = ptx_module_is ? intersection_shader_desc.entrypoint_name.c_str() : nullptr;

    return GetCachedProgramGroup(prog_group_desc);
}

OptixModule Pipeline::CreateNewModule(const std::string &ptx_filename)
{
    char   log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);

    if (ptx_filename.empty())
    {
        return nullptr;
    }

    OptixModule ptx_module;

    std::vector<char> ptxSource = ReadFile("./CrisOptixShader.dir/Debug/" + ptx_filename);
    OptixResult result = optixModuleCreate(
        m_Context,
        &m_ModuleCompileOptions,
        &m_PipelineCompileOptions,
        ptxSource.data(),
        ptxSource.size(),
        log,
        &sizeof_log,
        &ptx_module);
    if (result != OPTIX_SUCCESS)
    {
        Log::Error(log, true);
    }

    return ptx_module;
}

OptixModule Pipeline::GetCachedModule(const std::string &ptx_filename)
{
    OptixModule &ptx_module = m_ModuleCache[ptx_filename];
    if (ptx_module == nullptr) { ptx_module = CreateNewModule(ptx_filename); }
    return ptx_module;
}

void Pipeline::CreatePipeline()
{
    char   log[2048];
    size_t sizeof_log = sizeof(log);

    // Collect all program groups in plain vector
    std::vector<OptixProgramGroup> program_groups;
    std::transform(m_ProgramGroups.begin(), m_ProgramGroups.end(), std::back_inserter(program_groups), [](const auto &entry){return entry.second;});

    OptixResult result = optixPipelineCreate(
        m_Context, 
        &m_PipelineCompileOptions, 
        &m_PipelineLinkOptions,                  
        program_groups.data(), 
        program_groups.size(), 
        log,
        &sizeof_log, 
        &m_Handle);
    if (result != OPTIX_SUCCESS)
    {
        Log::Error(log, true);
    }
}
