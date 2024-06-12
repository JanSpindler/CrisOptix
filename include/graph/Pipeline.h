#pragma once

#include <optix.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <array>

struct OptixProgramGroupEntryKey
{
    OptixModule module              = nullptr;
    std::string entryname;

    OptixProgramGroupEntryKey() = default;
    OptixProgramGroupEntryKey(OptixModule _module, const char *_entryname) :
        module { _module },
        entryname { _entryname != nullptr ? _entryname : "" }
    {}

    bool operator == (const OptixProgramGroupEntryKey &other) const
    {
        return module == other.module &&
               entryname == other.entryname;
    }
};

struct OptixProgramGroupDescKey
{
    OptixProgramGroupKind kind;
    unsigned int flags          = 0;

    std::array<OptixProgramGroupEntryKey, 3> entryPoints;

    OptixProgramGroupDescKey() = default;
    OptixProgramGroupDescKey(const OptixProgramGroupDesc &desc) :
        kind { desc.kind },
        flags { desc.flags }
    {
        switch(kind)
        {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
                entryPoints[0] = { desc.raygen.module, desc.raygen.entryFunctionName };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_MISS:
                entryPoints[0] = { desc.miss.module, desc.miss.entryFunctionName };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
                entryPoints[0] = { desc.exception.module, desc.exception.entryFunctionName };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                entryPoints[0] = { desc.hitgroup.moduleCH, desc.hitgroup.entryFunctionNameCH };
                entryPoints[1] = { desc.hitgroup.moduleAH, desc.hitgroup.entryFunctionNameAH };
                entryPoints[2] = { desc.hitgroup.moduleIS, desc.hitgroup.entryFunctionNameIS };
                break;
            case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
                entryPoints[0] = { desc.callables.moduleDC, desc.callables.entryFunctionNameDC };
                entryPoints[1] = { desc.callables.moduleCC, desc.callables.entryFunctionNameCC };
                break;
        }
    }

    bool operator == (const OptixProgramGroupDescKey &other) const
    {
        return kind == other.kind &&
            flags == other.flags &&
            entryPoints[0] == other.entryPoints[0] &&
            entryPoints[1] == other.entryPoints[1] &&
            entryPoints[2] == other.entryPoints[2];
    }
};

template <typename T>
inline void hash_combine(std::size_t &s, const T &v)
{
    std::hash<T> h;
    s^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

namespace std 
{
    template<>
    struct hash<OptixProgramGroupEntryKey>
    {
        size_t operator()(const OptixProgramGroupEntryKey &v) const
        {
            size_t s = 0;
            hash_combine(s, v.module);
            hash_combine(s, v.entryname);
            return s;
        }
    };

    template<>
    struct hash<OptixProgramGroupDescKey>
    {
        size_t operator()(const OptixProgramGroupDescKey &v) const
        {
            size_t s = 0;
            hash_combine(s, v.kind);
            hash_combine(s, v.flags);
            for (const auto &e : v.entryPoints)
            {
                hash_combine(s, e);
            }
            return s;
        }
    };
} // end namespace std

struct ShaderEntryPointDesc
{
    std::string ptx_filename;
    std::string entrypoint_name;
};

class Pipeline
{
public:
    Pipeline(OptixDeviceContext context);
    ~Pipeline();

    OptixProgramGroup AddRaygenShader(const ShaderEntryPointDesc &raygen_shader_desc);
    OptixProgramGroup AddCallableShader(const ShaderEntryPointDesc &callable_shader_desc);
    OptixProgramGroup AddMissShader(const ShaderEntryPointDesc &miss_shader_desc);
    OptixProgramGroup AddTrianglesHitGroupShader(const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc);
    OptixProgramGroup AddProceduralHitGroupShader(const ShaderEntryPointDesc &intersection_shader_desc, const ShaderEntryPointDesc &closestHit_shader_desc, const ShaderEntryPointDesc &anyHit_shader_desc);

    void CreatePipeline();

    constexpr OptixPipeline GetHandle() const { return m_Handle; }

private:
    OptixModule CreateNewModule(const std::string &ptx_filename);
    OptixModule GetCachedModule(const std::string &ptx_filename);

    OptixProgramGroup CreateNewProgramGroup(const OptixProgramGroupDesc &prog_group_desc);
    OptixProgramGroup GetCachedProgramGroup(const OptixProgramGroupDesc &prog_group_desc);

private:
    OptixDeviceContext m_Context = nullptr;

    std::unordered_map<std::string, OptixModule> m_ModuleCache{};
    std::unordered_map<OptixProgramGroupDescKey, OptixProgramGroup> m_ProgramGroups{};

    OptixPipelineCompileOptions m_PipelineCompileOptions{};
    OptixModuleCompileOptions   m_ModuleCompileOptions{};
    OptixProgramGroupOptions    m_ProgramGroupOptions{};
    OptixPipelineLinkOptions    m_PipelineLinkOptions{};

    OptixPipeline m_Handle = nullptr;
};
