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

	// Exit if there is no primary hit
	if (!params.restir.restirGBuffers[pixelIdx].primaryInteraction.valid)
	{
		params.outputBuffer[pixelIdx] = glm::vec3(0.0f);
		return; 
	}

	// Init RNG
	const uint64_t seed = SampleTEA64(pixelIdx, params.random);
	PCG32 rng(seed);

	// Get prefix from stored res
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;

	// Exit if prefix is invalid
	if (!prefix.valid)
	{
		params.outputBuffer[pixelIdx] = glm::vec3(0.0f);
		return;
	}

	// Exit if prefix already terminated by NEE
	if (prefix.nee)
	{
		params.outputBuffer[pixelIdx] = prefix.f / prefix.p;
		return;
	}

	// Gen canonical suffix
	Reservoir<SuffixPath> suffixRes{};
	SuffixGen(suffixRes, prefix, rng);

	// Exit if suffix is invalid (Dont because we might find valid through resampling)
	const SuffixPath& suffix = suffixRes.sample;
	if (!suffix.valid) { return; }

	// Illuminate using suffix and prefix
	params.outputBuffer[pixelIdx] = prefix.f * suffix.f / (prefix.p * suffix.p);
}
