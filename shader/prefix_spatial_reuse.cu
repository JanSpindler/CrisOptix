#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <util/pixel_index.h>
#include <graph/restir/prefix_reuse.h>

__constant__ LaunchParams params;

static constexpr uint32_t WINDOW_RADIUS = 3; // 48 total neighbors
static constexpr uint32_t WINDOW_SIZE = 2 * WINDOW_RADIUS + 1;

static constexpr __forceinline__ __device__ glm::uvec2 SelectSpatialNeighbor(const glm::uvec2& pixelCoord, PCG32& rng)
{
	const uint32_t xCoord = (pixelCoord.x - WINDOW_RADIUS) + (rng.NextUint32() % WINDOW_SIZE);
	const uint32_t yCoord = (pixelCoord.y - WINDOW_RADIUS) + (rng.NextUint32() % WINDOW_SIZE);
	return glm::uvec2(xCoord, yCoord);
}

static __forceinline__ __device__ void PrefixSpatialReuse(const glm::uvec2& pixelCoord, PCG32& rng)
{
	// Assume: pixelCoord are valid

	// Get pixel index
	const uint32_t currPixelIdx = GetPixelIdx(pixelCoord, params);

	// Get current prefix
	Reservoir<PrefixPath>& currPrefixRes = params.restir.prefixReservoirs[2 * currPixelIdx + params.restir.frontBufferIdx];
	const PrefixPath& currPrefix = currPrefixRes.sample;
	if (!currPrefix.IsValid()) { return; }
	
	// Get canonical prefix
	const PrefixPath& canonPrefix = params.restir.canonicalPrefixes[currPixelIdx];
	if (!canonPrefix.IsValid()) { return; }

	// RIS with pairwise MIS weights
	uint32_t validNeighCount = 0;
	float canonicalWeight = 0.0f;

	for (uint32_t neighIdx = 0; neighIdx < params.restir.prefixSpatialCount; ++neighIdx)
	{
		// Select new neighbor
		const glm::uvec2 neighPixelCoord = SelectSpatialNeighbor(pixelCoord, rng);
		if (!IsPixelValid(neighPixelCoord, params)) { continue; }
		const uint32_t neighPixelIdx = GetPixelIdx(neighPixelCoord, params);

		// Get neighbor res and prefix
		const Reservoir<PrefixPath>& neighRes = params.restir.prefixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
		const PrefixPath& neighPrefix = neighRes.sample;
		if (!neighPrefix.IsValid() || neighPrefix.IsNee()) { continue; }

		// Get neighbor primary hit
		Interaction neighPrimaryInt{};
		TraceInteractionSeed(neighPrefix.primaryIntSeed, neighPrimaryInt, params);
		if (!neighPrimaryInt.valid) { continue; }
		++validNeighCount;

		//

	}
}

extern "C" __global__ void __raygen__prefix_spatial_reuse()
{
	//
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());
	const glm::uvec2 pixelCoord = glm::uvec2(launchIdx);

	// Exit if invalid launch idx
	if (launchIdx.x >= params.width || launchIdx.y >= params.height || launchIdx.z >= 1)
	{
		return;
	}

	// Get rng
	const uint32_t pixelIdx = launchIdx.y * launchDims.x + launchIdx.x;
	PCG32& rng = params.restir.restirGBuffers[pixelIdx].rng;

	// Spatial prefix reuse
	PrefixSpatialReuse(pixelCoord, rng);
}
