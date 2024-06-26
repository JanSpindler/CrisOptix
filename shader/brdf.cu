#include <graph/brdf.h>
#include <graph/trace.h>
#include <optix.h>
#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/Interaction.h>
#include <model/Material.h>
#include <texture_indirect_functions.h>

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

extern "C" __device__ BrdfResult __direct_callable__ggx(const SurfaceInteraction& interaction, const glm::vec3& outDir)
{
    // Get ggx data
    const MaterialSbtData* ggxData = *reinterpret_cast<const MaterialSbtData**>(optixGetSbtDataPointer());

    // Get values if possible from texture
    const glm::vec2 uv = interaction.uv;

    glm::vec3 diffColor = ggxData->diffColor;
    if (ggxData->hasDiffTex)
    {
        const float4 cuTex4 = tex2D<float4>(ggxData->diffTex, uv.x, uv.y);
        diffColor = glm::vec3(cuTex4.x, cuTex4.y, cuTex4.z);
    }

    // Calc diffuse brdf result
    const glm::vec3 diffBrdf = diffColor / PI;

    // Transform vectors into tangent space
    const glm::mat3 w2t = World2Tan(interaction.normal, interaction.tangent, glm::cross(interaction.normal, interaction.tangent));

    // Get vectors and scalars for equation
    const glm::vec3 l = w2t * outDir;
    const glm::vec3 v = w2t * -interaction.inRayDir;
    const glm::vec3 h = glm::normalize(l + v);
    const float a = ggxData->roughTangent;
    const float b = ggxData->roughBitangent;

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
    BrdfResult result{};
    //result.brdfResult = (diffBrdf + specBrdf) * clampedNdotL;
    result.brdfResult = diffBrdf;
    result.samplingPdf = 0.0f;
    return result;
}
