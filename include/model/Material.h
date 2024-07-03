#pragma once

#include <glm/glm.hpp>
#include <model/Texture.h>
#include <graph/brdf.h>
#include <graph/DeviceBuffer.h>
#include <graph/Pipeline.h>
#include <graph/ShaderBindingTable.h>

struct MaterialSbtData
{
    glm::vec3 diffColor;
    glm::vec3 specF0;
    float roughness;

    bool hasDiffTex;
    cudaTextureObject_t diffTex;

    bool hasSpecTex;
    cudaTextureObject_t specTex;

    bool hasRoughTex;
    cudaTextureObject_t roughTex;
};

class Material
{
public:
	Material(
        const glm::vec4& diffColor, 
        const glm::vec4& specColor, 
        const float roughness, 
        const Texture* diffTex,
        const Texture* specTex,
        const Texture* roughTex);
	
    void AddShader(Pipeline& pipeline, ShaderBindingTable& sbt);

    uint32_t GetEvalSbtIdx() const;

private:
	DeviceBuffer<MaterialSbtData> m_SbtDataBuf{};
    uint32_t m_EvalSbtIdx = 0;
};
