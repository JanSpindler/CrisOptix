#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <util/random.h>
#include <graph/LaunchParams.h>
#include <util/pixel_index.h>
#include <graph/restir/path_gen.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void SuffixGen(
	Reservoir<SuffixPath>& suffixRes,
	const PrefixPath& prefix,
	PCG32& rng)
{
	// Return invalid suffix if prefix already terminated into NEE before
	if (prefix.nee) { return; }

	const SuffixPath suffix = TraceSuffix(prefix, 8 - prefix.len, 8, rng, params);
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
	const uint64_t seed = SampleTEA64(pixelIdx, params.random);
	PCG32 rng(seed);

	// Get prefix from stored res
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;

	// Exit if prefix is invalid
	if (!prefix.valid) { return; }

	// Gen canonical suffix
	Reservoir<SuffixPath> suffixRes{};
	SuffixGen(suffixRes, prefix, rng);
}
