#include <cuda_runtime.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>

__constant__ LaunchParams params;

extern "C" __global__ void __raygen__prefix_store_entries()
{
	// Sanity check
	if (params.restir.gatherM <= 1) { return; }

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

	// Get prefix and suffix
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;
	const SuffixPath& suffix = params.restir.suffixReservoirs[pixelIdx].sample;

	// Store prefix entry in array
	// Prefix entry is only valid/usable if both prefix and suffix are valid
	params.restir.prefixEntries[pixelIdx] = PrefixEntry(prefix.valid && suffix.valid, prefix.lastInteraction.pos, pixelIdx);
}
