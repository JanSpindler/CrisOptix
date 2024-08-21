#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <util/pixel_index.h>
#include <graph/restir/prefix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void PrefixSpatialReuse(const glm::uvec2& pixelCoord, PCG32& rng)
{
	// Assume: pixelCoord are valid

	// Get current prefix
	const size_t currPixelIdx = GetPixelIdx(pixelCoord, params);
	Reservoir<PrefixPath>& currPrefixRes = params.restir.prefixReservoirs[currPixelIdx];
	const PrefixPath& currPrefix = currPrefixRes.sample;

	// Exit if current prefix is invalid or not fit for reuse
	if (!currPrefix.valid || currPrefix.len < params.restir.minPrefixLen) { return; }

	// Select random neighbor
	static constexpr uint32_t kernelRadius = 2;
	const uint32_t randX = rng.NextUint32() % (2 * kernelRadius + 1);
	const uint32_t randY = rng.NextUint32() % (2 * kernelRadius + 1);
	const glm::uvec2 neighPixelCoord = pixelCoord + glm::uvec2(randX, randY) - glm::uvec2(kernelRadius, kernelRadius);

	// Exit if neighbor is current pixel
	if (neighPixelCoord == pixelCoord) { return; }

	// Check if neighbor is on screen
	if (!IsPixelValid(neighPixelCoord, params)) { return; }

	// Get neighbor prefix reservoir
	const Reservoir<PrefixPath>& neighPrefixRes = params.restir.prefixReservoirs[GetPixelIdx(neighPixelCoord, params)];
	const PrefixPath& neighPrefix = neighPrefixRes.sample;

	// Prefix reuse
	const SurfaceInteraction& primaryInteraction = params.restir.restirGBuffers[currPixelIdx].primaryInteraction;
	PrefixReuse(currPrefixRes, neighPrefixRes, primaryInteraction, rng, params);
}

extern "C" __global__ void __raygen__prefix_spatial_reuse()
{
	// Sanity check
	if (!params.enableRestir || !params.restir.prefixEnableSpatial) { return; }

	//
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());
	const glm::uvec2 pixelCoord = glm::uvec2(launchIdx);

	// Exit if invalid launch idx
	if (launchIdx.x >= params.width || launchIdx.y >= params.height || launchIdx.z >= 1)
	{
		return;
	}

	// Init RNG
	// TODO: Other seed for each raygen
	const uint32_t pixelIdx = launchIdx.y * launchDims.x + launchIdx.x;
	const uint64_t seed = SampleTEA64(pixelIdx, params.random);
	PCG32 rng(seed);

	//
	PrefixSpatialReuse(pixelCoord, rng);
}
