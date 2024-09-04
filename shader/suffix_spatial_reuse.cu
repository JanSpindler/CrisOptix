#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>
#include <graph/restir/suffix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void SuffixSpatialReuse(const glm::uvec2& pixelCoord, PCG32& rng)
{
	// Assume: pixelCoord are valid

	// Get current prefix
	const size_t currPixelIdx = GetPixelIdx(pixelCoord, params);
	const PrefixPath& prefix = params.restir.prefixReservoirs[currPixelIdx].sample;
	
	// Exit if current prefix is invalid
	if (!prefix.IsValid()) { return; }

	// Get current suffix
	Reservoir<SuffixPath>& currSuffixRes = params.restir.suffixReservoirs[currPixelIdx];
	const SuffixPath& currSuffix = currSuffixRes.sample;

	// Exit if current suffix 
	// TODO: Should this be the case?
	if (!currSuffix.IsValid()) { return; }

	// Select random neighbor
	// TODO: Kernel radius as parameter
	static constexpr uint32_t kernelRadius = 2;
	const uint32_t randX = rng.NextUint32() % (2 * kernelRadius + 1);
	const uint32_t randY = rng.NextUint32() % (2 * kernelRadius + 1);
	const glm::uvec2 neighPixelCoord = pixelCoord + glm::uvec2(randX, randY) - glm::uvec2(kernelRadius, kernelRadius);

	// Exit if neighbor is current pixel
	if (neighPixelCoord == pixelCoord) { return; }

	// Check if neighbor is on screen
	if (!IsPixelValid(neighPixelCoord, params)) { return; }

	// Get neighbor suffix reservoir
	const Reservoir<SuffixPath>& neighSuffixRes = params.restir.suffixReservoirs[GetPixelIdx(neighPixelCoord, params)];

	// Suffix reuse
	SuffixReuse(currSuffixRes, neighSuffixRes, prefix, rng, params);
}

extern "C" __global__ void __raygen__suffix_spatial_reuse()
{
	// Sanity check
	if (!params.enableRestir || !params.restir.suffixEnableSpatial) { return; }

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
	PCG32& rng = params.restir.restirGBuffers[pixelIdx].rng;

	// Spatial suffix reuse
	SuffixSpatialReuse(pixelCoord, rng);
}
