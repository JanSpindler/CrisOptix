#pragma once

#include <glm/glm.hpp>
#include <model/Texture.h>
#include <graph/brdf.h>
#include <graph/DeviceBuffer.h>
#include <graph/Pipeline.h>
#include <graph/ShaderBindingTable.h>

enum class SpecTexUsage
{
    Color,
    OccRoughMetal
};

struct MaterialSbtData
{
    glm::vec3 diffColor;
    glm::vec3 specF0;
    glm::vec3 emissiveColor;
    float roughness;

    bool hasDiffTex;
    cudaTextureObject_t diffTex;

    bool hasSpecTex;
    cudaTextureObject_t specTex;

    bool hasRoughTex;
    cudaTextureObject_t roughTex;

    SpecTexUsage specTexUsage;
};

class Material
{
public:
	Material(
        const glm::vec4& diffColor, 
        const glm::vec4& specColor, 
        const glm::vec4& emissiveColor,
        const float roughness, 
        const Texture* diffTex,
        const Texture* specTex,
        const Texture* roughTex, 
        const SpecTexUsage specTexUsage);
	
    void AddShader(Pipeline& pipeline, ShaderBindingTable& sbt);

    uint32_t GetEvalSbtIdx() const;
    uint32_t GetSampleSbtIdx() const;
    glm::vec3 GetEmissiveColor() const;

private:
	DeviceBuffer<MaterialSbtData> m_SbtDataBuf{};
    uint32_t m_EvalSbtIdx = 0;
    uint32_t m_SampleSbtIdx = 0;
    const glm::vec3 m_EmissiveColor;
};
