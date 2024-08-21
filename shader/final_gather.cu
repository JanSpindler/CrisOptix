#include <cuda_runtime.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>

__constant__ LaunchParams params;

extern "C" __global__ void __raygen__final_gather()
{
	//
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());
	const glm::uvec2 pixelCoord = glm::uvec2(launchIdx);
	const size_t pixelIdx = GetPixelIdx(pixelCoord, params);

	// Exit if invalid launch idx
	if (launchIdx.x >= params.width || launchIdx.y >= params.height || launchIdx.z >= 1)
	{
		return;
	}

	// Init RNG
	const uint64_t seed = SampleTEA64(pixelIdx, params.random);
	PCG32 rng(seed);

	//
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;
	const SuffixPath& suffix = params.restir.suffixReservoirs[pixelIdx].sample;

	glm::vec3 outputRadiance(0.0f);
	if (prefix.valid)
	{
		if (suffix.valid)
		{
			outputRadiance = (prefix.f * suffix.f) / (prefix.p * suffix.p);
		}
		else if (prefix.nee)
		{
			outputRadiance = prefix.f / prefix.p;
		}
	}

	params.outputBuffer[pixelIdx] = outputRadiance;
}
