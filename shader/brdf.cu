#include <graph/brdf.h>
#include <graph/trace.h>
#include <optix.h>
#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/Interaction.h>
#include <model/Material.h>
#include <texture_indirect_functions.h>
#include <util/random.h>

static constexpr float PI = 3.14159265358979323846264f;

static __forceinline__ __device__ float P2(const float x)
{
    return x * x;
}

static __forceinline__ __device__ float P4(const float x)
{
    return P2(P2(x));
}

static __forceinline__ __device__ float CDot(const glm::vec3& a, const glm::vec3& b)
{
    return glm::clamp(glm::dot(a, b), 0.0f, 1.0f);
}

static __forceinline__ __device__ float FresnelSchlick(const float F0, const float VdotH)
{
    return F0 + (1.0f - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

static __forceinline__ __device__ glm::vec3 FresnelSchlick(const glm::vec3& F0, const float VdotH)
{
    return F0 + (glm::vec3(1.0f) - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

static __forceinline__ __device__ float D(const glm::vec3& h, const float ax, const float ay)
{
    if (h.z < 0) { return 0.0f; }
    return 1 / (PI * ax * ay) * 1 / P2((P2(h.x / ax) + P2(h.y / ay) + P2(h.z)));
}

static __forceinline__ __device__ float Lambda(const glm::vec3& v, float ax, float ay)
{
    return (-1 + sqrt(1 + (P2(v.x * ax) + P2(v.y * ay)) / P2(v.z))) / 2;
}

static __forceinline__ __device__ float G2(const glm::vec3& to_v, const glm::vec3& to_l, const glm::vec3& h, float ax, float ay)
{
    if (dot(to_v, h) <= 0 || dot(to_l, h) <= 0) { return 0.0f; }
    return 1 / (1 + Lambda(to_v, ax, ay) + Lambda(to_l, ax, ay));
}

static __forceinline__ __device__ glm::mat3 World2Tan(const glm::vec3& n, const glm::vec3& tan, const glm::vec3& bitan)
{
    // tan-to-worldspace is matrix with columns tan, bitan, n (s.t. x is mapped
    // to the tangent, y to the bitangent, z to the normal), we want the
    // inverse, which is just the transpose since the matrix is unitary (all
    // columns are orthogonal, obviously).
    return glm::transpose(glm::mat3(tan, bitan, n));
}

static __forceinline__ __device__ glm::vec3 GetFromTexIfPossible(
    const bool hasTex, 
    const glm::vec3& defaultVal,
    const glm::vec2 uv,
    const cudaTextureObject_t tex)
{
    if (hasTex)
    {
        const float4 cuTex4 = tex2D<float4>(tex, uv.x, uv.y);
        return glm::vec3(cuTex4.x, cuTex4.y, cuTex4.z);
    }
    else
    {
        return defaultVal;
    }
}

static __forceinline__ __device__ float GetFromTexIfPossible(
    const bool hasTex,
    const float defaultVal,
    const glm::vec2 uv,
    const cudaTextureObject_t tex)
{
    if (hasTex)
    {
        const float4 cuTex4 = tex2D<float4>(tex, uv.x, uv.y);
        return cuTex4.x;
    }
    else
    {
        return defaultVal;
    }
}

static __forceinline__ __device__ float D_GGX(const float NdotH, const float roughness)
{
    float a2 = roughness * roughness;
    float d = (NdotH * a2 - NdotH) * NdotH + 1.0f;
    return a2 / (PI * d * d);
}

static __forceinline__ __device__ float V_SmithJointGGX(float NdotL, float NdotV, float roughness)
{
    float a2 = roughness * roughness;
    float lambdaV = NdotL * glm::sqrt(NdotV * NdotV * (1 - a2) + a2);
    float lambdaL = NdotV * glm::sqrt(NdotL * NdotL * (1 - a2) + a2);
    return 0.5f / (lambdaV + lambdaL);
}


extern "C" __device__ BrdfEvalResult __direct_callable__ggx_eval(const SurfaceInteraction& interaction, const glm::vec3& outDir)
{
    // Get ggx data
    const MaterialSbtData* ggxData = *reinterpret_cast<const MaterialSbtData**>(optixGetSbtDataPointer());

    // Get values if possible from texture
    const glm::vec2 uv = interaction.uv;

    // Get diff color
    const glm::vec3 diffColor = GetFromTexIfPossible(ggxData->hasDiffTex, ggxData->diffColor, uv, ggxData->diffTex);

    // Get specular info
    glm::vec3 specF0 = GetFromTexIfPossible(ggxData->hasSpecTex, ggxData->specF0, uv, ggxData->specTex);
    float roughness = GetFromTexIfPossible(ggxData->hasRoughTex, ggxData->roughness, uv, ggxData->roughTex);
    float metal = 0.0;
    if (ggxData->specTexUsage == SpecTexUsage::OccRoughMetal)
    {
        roughness = specF0.g;
        metal = specF0.b;
        specF0 = glm::vec3(1.0f);
    }

    // Apply metal
    specF0 = glm::mix(diffColor, specF0, metal);

    //
    BrdfEvalResult result{};

    //
    const glm::vec3 lightDir = outDir;
    const glm::vec3 viewDir = -interaction.inRayDir;
    const glm::vec3 normal = interaction.normal;

    const float nDotV = glm::dot(normal, viewDir);
    const float nDotL = glm::dot(normal, lightDir);
    if (nDotV <= 0.0f || nDotL < 0.0f)
    {
        result.brdfResult = glm::vec3(0.0f);
        result.emission = glm::vec3(0.0f);
        result.samplingPdf = 0.0f;
        result.roughness = 0.0f;
        return result;
    }

    // Calc diffuse brdf result
    const glm::vec3 diffBrdf = diffColor / PI;

    // Compute specular brdf
    glm::vec3 specBrdf = glm::vec3(0.0f);

    const float diffProb = glm::dot(diffColor, glm::vec3(1)) / (glm::dot(diffColor, glm::vec3(1)) + glm::dot(specF0, glm::vec3(1)));
    const glm::vec3 halfway = glm::normalize(lightDir + viewDir);
    const float nDotH = glm::dot(halfway, normal);
    const float lDotH = glm::dot(halfway, lightDir);
    // Only compute specular component if specular_f0 is not zero!
    if (glm::dot(specF0, specF0) > 1e-6)
    {
        // Normal distribution
        const float d = D_GGX(nDotH, roughness);

        // Visibility
        const float v = V_SmithJointGGX(nDotL, nDotV, roughness);

        // Fresnel
        const glm::vec3 f = FresnelSchlick(specF0, lDotH);

        specBrdf = d * v * f;
    }

    // Result
    result.brdfResult = (diffBrdf + specBrdf) * nDotL;
    result.samplingPdf = diffProb * nDotL / PI + (1.0f - diffProb) * D_GGX(nDotH, roughness) * nDotH / (4.0f * lDotH);
    result.emission = ggxData->emissiveColor;
    result.roughness = roughness;
    return result;
}

static __forceinline__ __device__ glm::mat3 ComputeLocalFrame(const glm::vec3& localZ)
{
    float x = localZ.x;
    float y = localZ.y;
    float z = localZ.z;
    float sz = (z >= 0) ? 1 : -1;
    float a = 1 / (sz + z);
    float ya = y * a;
    float b = x * ya;
    float c = x * sz;

    glm::vec3 localX = glm::vec3(c * x * a - 1, sz * b, c);
    glm::vec3 localY = glm::vec3(b, y * ya - sz, y);

    glm::mat3 frame(1.0f);
    // Set columns of matrix
    frame[0] = localX;
    frame[1] = localY;
    frame[2] = localZ;
    return frame;
}

extern "C" __device__ BrdfSampleResult __direct_callable__ggx_sample(const SurfaceInteraction& interaction, PCG32& rng)
{
    // Get ggx data
    const MaterialSbtData* ggxData = *reinterpret_cast<const MaterialSbtData**>(optixGetSbtDataPointer());

    // Get values if possible from texture
    const glm::vec2 uv = interaction.uv;

    // Diffuse
    const glm::vec3 diffColor = GetFromTexIfPossible(ggxData->hasDiffTex, ggxData->diffColor, uv, ggxData->diffTex);

    // Get specular info
    glm::vec3 specF0 = GetFromTexIfPossible(ggxData->hasSpecTex, ggxData->specF0, uv, ggxData->specTex);
    float roughness = GetFromTexIfPossible(ggxData->hasRoughTex, ggxData->roughness, uv, ggxData->roughTex);
    float metal = 0.0;
    if (ggxData->specTexUsage == SpecTexUsage::OccRoughMetal)
    {
        roughness = specF0.g;
        metal = specF0.b;
        specF0 = glm::vec3(1.0f);
    }

    // Apply metal
    specF0 = glm::mix(diffColor, specF0, metal);

    // Get info from interaction
    const glm::vec3 normal = interaction.normal;
    const glm::vec3 viewDir = -interaction.inRayDir;

    // Check if valid
    BrdfSampleResult result{};
    result.roughness = roughness;

    const float nDotV = glm::dot(normal, viewDir);
    if (nDotV <= 0)
    {
        result.outDir = glm::vec3(0.0f);
        result.brdfVal = glm::vec3(0.0f);
        result.samplingPdf = 0.0f;
        return result;
    }

    //
    const glm::mat3 localFrame = ComputeLocalFrame(normal);

    //
    const float diffProb = glm::dot(diffColor, glm::vec3(1)) / (glm::dot(diffColor, glm::vec3(1)) + glm::dot(specF0, glm::vec3(1)));

    //
    if (rng.NextFloat() < diffProb)
    {
        // Diffuse
        // Malleys method
        const float phi = rng.NextFloat() * 2.0f * PI;
        const float r = glm::sqrt(rng.NextFloat());

        const float x = r * glm::cos(phi);
        const float y = r * glm::sin(phi);
        const float z = glm::sqrt(glm::clamp<float>(1.0f - P2(x) - P2(y), 0.0f, 1.0f));

        result.outDir = localFrame * glm::vec3(x, y, z);
    }
    else
    {
        // Specular
        //
        const float u = rng.NextFloat();

        // Sample z using iCDF
        const float microNormalZ = glm::sqrt(glm::clamp<float>((1 - u) / (1 + (P2(roughness) - 1) * u), 0, 1));
        const float microNormalR = glm::sqrt(glm::clamp<float>(1 - P2(microNormalZ), 0, 1));

        // Pick phi uniformly
        const float phi = rng.NextFloat() * 2.0f * PI;
        glm::vec3 microNormal(microNormalR * glm::cos(phi), microNormalR * glm::sin(phi), microNormalZ);

        // Out dir
        result.outDir = glm::reflect(interaction.inRayDir, localFrame * microNormal);

        //
        if (glm::dot(result.outDir, normal) < 0.0f)
        {
            result.outDir = glm::vec3(0.0f);
            result.brdfVal = glm::vec3(0.0f);
            result.samplingPdf = 0.0f;
            return result;
        }
    }

    // Compute brdf
    // Compute diffuse brdf
    const glm::vec3 diffBrdf = diffColor / PI;

    // Specular brdf
    const glm::vec3 lightDir = result.outDir;
    const float nDotL = glm::dot(normal, lightDir);
    const glm::vec3 halfway = glm::normalize(lightDir + viewDir);
    const float nDotH = glm::dot(halfway, normal);
    const float lDotH = glm::dot(halfway, lightDir);

    // Compute specular BSDF
    glm::vec3 specBrdf = glm::vec3(0.0f);
    // Only compute specular component if specular_f0 is not zero!
    if (glm::dot(specF0, specF0) > 1e-6)
    {
        // Normal distribution
        const float d = D_GGX(nDotH, roughness);

        // Visibility
        const float v = V_SmithJointGGX(nDotL, nDotV, roughness);

        // Fresnel
        const glm::vec3 f = FresnelSchlick(specF0, lDotH);

        // not sure why /(4*NdotL*NdotV) is missing in eval_BSDF, but we're
        // doing that here too.
        specBrdf = d * v * f;
    }

    // calculate pdf according to one-sample balance heuristic.
    result.outDir = lightDir;
    result.samplingPdf = diffProb * nDotL / PI + (1.0f - diffProb) * D_GGX(nDotH, roughness) * nDotH / (4.0f * lDotH);
    result.brdfVal = (diffBrdf + specBrdf) * nDotL;
    return result;
}
