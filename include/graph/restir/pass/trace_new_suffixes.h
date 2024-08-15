#pragma once

#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/struct/Reconnection.h>

static __forceinline__ __device__ void TraceNewSuffixes(
    const size_t pixelIdx,
    const PrefixPath& prefix,
    SuffixPath& suffix,
    Reconnection& recon,
    PCG32& rng,
    const LaunchParams& params)
{
    // Exit if prefix interaction invalid
    if (!prefix.valid) { return; }

    // Generate reconnection
    // Sample brdf at last prefix vertex
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
    GenSuffix(suffix, interaction.pos, recon.pos1Brdf.outDir, 8 - prefix.len, 0.5f, 8, rng, params);
}
