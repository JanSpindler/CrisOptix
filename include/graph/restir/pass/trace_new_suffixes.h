#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/struct/Reconnection.h>

static __forceinline__ __device__ void TraceNewSuffixes(
    const size_t pixelIdx,
    const PrefixPath& prefix,
    Reservoir<SuffixPath>& suffixRes,
    Reconnection& recon,
    PCG32& rng,
    const LaunchParams& params)
{
    // Exit if prefix interaction invalid
    if (!prefix.valid) { return; }

    // Generate reconnection
    SuffixPath suffix{};

    // Sample brdf at last prefix vertex
    if (rng.NextFloat() > params.neeProb)
    {
        recon.pos0Brdf = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
            prefix.lastInteraction.meshSbtData->sampleMaterialSbtIdx,
            prefix.lastInteraction,
            rng);
        if (recon.pos0Brdf.samplingPdf <= 0.0f)
        {
            suffix.valid = false;
            return;
        }

        // Sample first vertex of suffix
        SurfaceInteraction interaction{};
        TraceWithDataPointer<SurfaceInteraction>(
            params.traversableHandle,
            prefix.lastInteraction.pos,
            recon.pos0Brdf.outDir,
            1e-3f,
            1e16f,
            params.surfaceTraceParams,
            &interaction);
        if (!interaction.valid)
        {
            suffix.valid = false;
            return;
        }

        // Sample out dir
        recon.pos1Brdf = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
            interaction.meshSbtData->sampleMaterialSbtIdx,
            interaction,
            rng);
        if (recon.pos1Brdf.samplingPdf <= 0.0f)
        {
            suffix.valid = false;
            return;
        }

        // Gen path
        GenSuffix(suffix, interaction.pos, recon.pos1Brdf.outDir, 8 - prefix.len, 8, rng, params);
    }
    // First suffix vert is nee
    else
    {
        suffix.len = 0;

        // Sample light source
        const EmitterSample emitterSample = SampleEmitter(rng, params.emitterTable);
        const glm::vec3 lightDir = glm::normalize(emitterSample.pos - prefix.lastInteraction.pos);
        const float distance = glm::length(emitterSample.pos - prefix.lastInteraction.pos);

        // Cast shadow ray
        const bool occluded = TraceOcclusion(
            params.traversableHandle,
            prefix.lastInteraction.pos,
            lightDir,
            1e-3f,
            distance,
            params.occlusionTraceParams);

        // If emitter is occluded -> skip
        if (occluded)
        {
            suffix.valid = false;
            suffix.p = 0.0f;
            suffix.throughput = glm::vec3(0.0f);
        }
        // If emitter is not occluded -> end NEE
        else
        {
            // Calc brdf
            const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
                prefix.lastInteraction.meshSbtData->evalMaterialSbtIdx,
                prefix.lastInteraction,
                lightDir);
            suffix.throughput = emitterSample.color;
            suffix.valid = true;
            suffix.p = emitterSample.p;

            recon.pos0Brdf.outDir = lightDir;
            recon.pos0Brdf.diffuse = true;
            recon.pos0Brdf.roughness = brdfEvalResult.roughness;
            recon.pos0Brdf.samplingPdf = emitterSample.p;
            recon.pos0Brdf.weight = brdfEvalResult.brdfResult;

            recon.pos1Brdf.outDir = glm::vec3(0.0f);
            recon.pos1Brdf.diffuse = true;
            recon.pos1Brdf.roughness = 1.0f;
            recon.pos1Brdf.samplingPdf = 1.0f;
            recon.pos1Brdf.weight = glm::vec3(1.0f);
        }
    }

    // Stream canonical suffix into res
    suffixRes.Update(suffix, suffix.GetWeight() * recon.GetWeight(), suffix.throughput, rng);
}
