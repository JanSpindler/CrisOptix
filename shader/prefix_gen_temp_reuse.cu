#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>
#include <util/pixel_index.h>
#include <graph/restir/ris_helper.h>
#include <graph/restir/prefix_reuse.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ SurfaceInteraction PrefixGen(
	Reservoir<PrefixPath>& prefixRes,
	const glm::vec3& origin,
	const glm::vec3& dir,
	PCG32& rng)
{
	// Trace new canonical prefix
	SurfaceInteraction primaryInteraction{};
	const PrefixPath prefix = TracePrefix(origin, dir, params.restir.minPrefixLen, 8, primaryInteraction, rng, params);

	// Do not store if not valid
	if (!prefix.valid) { return {}; }

	// Stream into res
	const float pHat = GetLuminance(prefix.f);
	const float risWeight = pHat / prefix.p;
	prefixRes.Update(prefix, risWeight, rng);

	//
	return primaryInteraction;
}

static __forceinline__ __device__ void PrefixTempReuse(
	Reservoir<PrefixPath>& prefixRes,
	const glm::uvec2& pixelCoord,
	const glm::uvec2& prevPixelCoord,
	const SurfaceInteraction& primaryInteraction,
	PCG32& rng)
{
	// Exit if current prefix is invalid
	const PrefixPath& currPrefix = prefixRes.sample;
	if (!currPrefix.valid) { return; }

	// Exit if primary interaction is invalid
	if (!primaryInteraction.valid) { return; }

	// Get previous prefix res
	const size_t prevPixelIdx = GetPixelIdx(prevPixelCoord, params);
	const Reservoir<PrefixPath>& prevPrefixRes = params.restir.prefixReservoirs[prevPixelIdx];

	// Prefix reuse
	PrefixReuse(prefixRes, prevPrefixRes, primaryInteraction, rng, params);
}

extern "C" __global__ void __raygen__prefix_gen_temp_reuse()
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

	// Init radiance with 0
	glm::vec3 outputRadiance(0.0f);

	// Spawn camera ray
	glm::vec3 origin(0.0f);
	glm::vec3 dir(0.0f);
	glm::vec2 uv = (glm::vec2(launchIdx) + rng.Next2d()) / glm::vec2(params.width, params.height);
	uv = 2.0f * uv - 1.0f; // [0, 1] -> [-1, 1]
	SpawnCameraRay(params.cameraData, uv, origin, dir);

	// If restir
	if (params.enableRestir)
	{
		// Generate canonical prefix and stream into new reservoir
		Reservoir<PrefixPath> prefixRes{};
		const SurfaceInteraction primaryInteraction = PrefixGen(prefixRes, origin, dir, rng);

		// Get motion vector
		const size_t pixelIdx = GetPixelIdx(pixelCoord, params);
		const glm::vec2 motionVector = params.motionVectors[pixelIdx];

		// Calc prev pixel coord
		const glm::uvec2 prevPixelCoord = glm::uvec2(glm::vec2(pixelCoord) + glm::vec2(0.5f) + motionVector);

		// Perform temporal prefix reuse if requested and if previous pixel is valid
		if (params.restir.prefixEnableTemporal && IsPixelValid(prevPixelCoord, params))
		{
			PrefixTempReuse(prefixRes, pixelCoord, prevPixelCoord, primaryInteraction, rng);
		}

		// If the resampled prefix is valid
		if (prefixRes.sample.valid)
		{
			// Store prefix reservoir and restir g buffer
			params.restir.prefixReservoirs[GetPixelIdx(pixelCoord, params)] = prefixRes;
			params.restir.restirGBuffers[pixelIdx] = RestirGBuffer(primaryInteraction, prevPixelCoord);
			outputRadiance = prefixRes.sample.f / prefixRes.sample.p;
		}
	}
	// If not restir
	else
	{
		// Perform normal path tracing
		outputRadiance = TraceCompletePath(origin, dir, 8, 8, rng, params);
	}

	// Store radiance output
	// TODO: Move accumulation is separate pass at the very end
	if (params.enableAccum)
	{
		const glm::vec3 oldVal = params.outputBuffer[pixelIdx];
		const float blendFactor = 1.0f / static_cast<float>(params.frameIdx + 1);
		params.outputBuffer[pixelIdx] = blendFactor * outputRadiance + (1.0f - blendFactor) * oldVal;
	}
	else
	{
		params.outputBuffer[pixelIdx] = outputRadiance;
	}
}
