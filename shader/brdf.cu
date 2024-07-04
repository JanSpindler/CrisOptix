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

static constexpr __device__ float P2(const float x)
{
    return x * x;
}

static constexpr __device__ float P4(const float x)
{
    return P2(P2(x));
}

static constexpr __device__ float CDot(const glm::vec3& a, const glm::vec3& b)
{
    return glm::clamp(glm::dot(a, b), 0.0f, 1.0f);
}

static constexpr __device__ float FresnelSchlick(const float F0, const float VdotH)
{
    return F0 + (1.0f - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

static constexpr __device__ glm::vec3 FresnelSchlick(const glm::vec3& F0, const float VdotH)
{
    return F0 + (glm::vec3(1.0f) - F0) * glm::pow(glm::max(0.0f, 1.0f - VdotH), 5.0f);
}

static constexpr __device__ float D(const glm::vec3& h, const float ax, const float ay)
{
    if (h.z < 0) { return 0.0f; }
    return 1 / (PI * ax * ay) * 1 / P2((P2(h.x / ax) + P2(h.y / ay) + P2(h.z)));
}

static constexpr __device__ float Lambda(const glm::vec3& v, float ax, float ay)
{
    return (-1 + sqrt(1 + (P2(v.x * ax) + P2(v.y * ay)) / P2(v.z))) / 2;
}

static constexpr __device__ float G2(const glm::vec3& to_v, const glm::vec3& to_l, const glm::vec3& h, float ax, float ay)
{
    if (dot(to_v, h) <= 0 || dot(to_l, h) <= 0) { return 0.0f; }
    return 1 / (1 + Lambda(to_v, ax, ay) + Lambda(to_l, ax, ay));
}

static constexpr __device__ glm::mat3 World2Tan(const glm::vec3& n, const glm::vec3& tan, const glm::vec3& bitan)
{
    // tan-to-worldspace is matrix with columns tan, bitan, n (s.t. x is mapped
    // to the tangent, y to the bitangent, z to the normal), we want the
    // inverse, which is just the transpose since the matrix is unitary (all
    // columns are orthogonal, obviously).
    return glm::transpose(glm::mat3(tan, bitan, n));
}

static constexpr __device__ glm::vec3 GetFromTexIfPossible(
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

static constexpr __device__ float GetFromTexIfPossible(
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

static constexpr __device__ float D_GGX(const float NdotH, const float roughness)
{
    float a2 = roughness * roughness;
    float d = (NdotH * a2 - NdotH) * NdotH + 1.0f;
    return a2 / (PI * d * d);
}

static constexpr __device__ float V_SmithJointGGX(float NdotL, float NdotV, float roughness)
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

    const glm::vec3 diffColor = GetFromTexIfPossible(ggxData->hasDiffTex, ggxData->diffColor, uv, ggxData->diffTex);
    const glm::vec3 specF0 = GetFromTexIfPossible(ggxData->hasSpecTex, ggxData->specF0, uv, ggxData->specTex);
    const float roughness = GetFromTexIfPossible(ggxData->hasRoughTex, ggxData->roughness, uv, ggxData->roughTex);

    // Calc diffuse brdf result
    const glm::vec3 diffBrdf = diffColor / PI;

    // Transform vectors into tangent space
    const glm::mat3 w2t = World2Tan(interaction.normal, interaction.tangent, glm::cross(interaction.normal, interaction.tangent));

    // Get vectors and scalars for equation
    const glm::vec3 l = w2t * outDir;
    const glm::vec3 v = w2t * -interaction.inRayDir;
    const glm::vec3 h = glm::normalize(l + v);
    const float a = ggxData->roughness;
    const float b = ggxData->roughness;

    // Calc specular brdf
    const glm::vec3 specBrdf = FresnelSchlick(ggxData->specF0, glm::dot(h, v))
        * D(h, a, b)
        * G2(v, l, h, a, b)
        / (4.0f * l.z * v.z);

    // <n, l>
    const float clampedNdotL = glm::max<float>(
        0.0f,
        glm::dot(outDir, interaction.normal) * -glm::sign(glm::dot(interaction.inRayDir, interaction.normal)));

    // Result
    BrdfEvalResult result{};
    result.brdfResult = (diffBrdf + specBrdf) * clampedNdotL;
    result.samplingPdf = 1.0f / (2.0f * PI); // 1 over area(unit hemisphere)
    result.emission = ggxData->emissiveColor;
    return result;
}

static constexpr __device__ glm::mat3 ComputeLocalFrame(const glm::vec3& localZ)
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

    const glm::vec3 diffColor = GetFromTexIfPossible(ggxData->hasDiffTex, ggxData->diffColor, uv, ggxData->diffTex);
    const glm::vec3 specF0 = GetFromTexIfPossible(ggxData->hasSpecTex, ggxData->specF0, uv, ggxData->specTex);
    const float roughness = GetFromTexIfPossible(ggxData->hasRoughTex, ggxData->roughness, uv, ggxData->roughTex);

    // Get info from interaction
    const glm::vec3 normal = interaction.normal;
    const glm::vec3 viewDir = -interaction.inRayDir;

    // Check if valid
    BrdfSampleResult result{};
    const float nDotV = glm::dot(normal, viewDir);
    if (nDotV <= 0)
    {
        result.outDir = glm::vec3(0.0f);
        result.weight = glm::vec3(0.0f);
        result.samplingPdf = 0.0f;
        return result;
    }

    //
    const glm::mat3 localFrame = ComputeLocalFrame(normal);

    //
    const float diffProb = glm::dot(diffColor, glm::vec3(1)) / (glm::dot(diffColor, glm::vec3(1)) + glm::dot(specF0, glm::vec3(1)));

    //
    bool sampleInvalid = false;

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

        //
        if (glm::dot(result.outDir, normal) < 0.0f)
        {
            sampleInvalid = true;
        }
    }

    // Invalid sample
    if (sampleInvalid)
    {
        result.outDir = glm::vec3(0.0f);
        result.weight = glm::vec3(0.0f);
        result.samplingPdf = 0.0f;
        return result;
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
    result.weight = (diffBrdf + specBrdf) * nDotL / result.samplingPdf;
    return result;
}
