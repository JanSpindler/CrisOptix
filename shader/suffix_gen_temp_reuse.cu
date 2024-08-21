#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <util/random.h>
#include <graph/LaunchParams.h>
#include <util/pixel_index.h>
#include <graph/restir/path_gen.h>
#include <graph/restir/suffix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void SuffixGen(
	Reservoir<SuffixPath>& suffixRes,
	const PrefixPath& prefix,
	PCG32& rng)
{
	// Assume: Prefix is valid and can be used to generate a suffix

	// Trace canonical suffix
	const SuffixPath suffix = TraceSuffix(prefix, 8 - prefix.len, 8, rng, params);

	// Do not store if not valid
	if (!suffix.valid) { return; }

	// Stream into reservoir
	const float pHat = GetLuminance(suffix.f);
	const float risWeight = pHat / suffix.p;
	suffixRes.Update(suffix, risWeight, rng);
}

static __forceinline__ __device__ void SuffixTempReuse(
	Reservoir<SuffixPath>& suffixRes,
	const PrefixPath& prefix,
	const glm::uvec2& prevPixelCoord,
	PCG32& rng)
{
	// Assume: Prefix is valid

	// Exit if prev pixel is invalid
	if (!IsPixelValid(prevPixelCoord, params)) { return; }

	// Get prev pixel suffix reservoir
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);
	const Reservoir<SuffixPath>& prevSuffixRes = params.restir.suffixReservoirs[prevPixelIdx];

	// Exit if prev suffix is invalid
	if (!prevSuffixRes.sample.valid) { return; }

	// Reuse prev suffix
	SuffixReuse(suffixRes, prevSuffixRes, prefix, rng, params);
}

extern "C" __global__ void __raygen__suffix_gen_temp_reuse()
{
	// Sanity check
	if (!params.enableRestir) { return; }

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

	// Get prefix from stored res
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;

	// Exit if prefix is invalid
	if (!prefix.valid || prefix.nee || !prefix.lastInteraction.valid) { return; }

	// Gen canonical suffix
	Reservoir<SuffixPath> suffixRes{};
	SuffixGen(suffixRes, prefix, rng);

	// Temporal suffix reuse
	if (params.restir.suffixEnableTemporal)
	{
		SuffixTempReuse(suffixRes, prefix, params.restir.restirGBuffers[pixelIdx].prevPixelCoord, rng);
	}

	// Store suffix res
	params.restir.suffixReservoirs[pixelIdx] = suffixRes;
}
