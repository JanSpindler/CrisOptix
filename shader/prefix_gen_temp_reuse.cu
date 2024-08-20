#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>

__constant__ LaunchParams params;

static __forceinline__ __device__ void PrefixGen(
	Reservoir<PrefixPath>& prefixRes,
	const glm::uvec2& pixelCoord,
	const glm::vec3& origin,
	const glm::vec3& dir,
	PCG32& rng)
{
	const PrefixPath prefix = TracePrefix(origin, dir, params.restir.minPrefixLen, 8, rng, params);
	if (!prefix.valid) { return; }

	const float pHat = GetLuminance(prefix.f);
	const float risWeight = pHat / prefix.p;
	const glm::vec3 fOverP = prefix.f / prefix.p;
	prefixRes.Update(prefix, risWeight, fOverP, rng);
}

static __forceinline__ __device__ void PrefixTempReuse(Reservoir<PrefixPath>& prefixRes)
{
}

extern "C" __global__ void __raygen__prefix_gen_temp_reuse()
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

	// Init RNG
	const uint32_t pixelIdx = launchIdx.y * launchDims.x + launchIdx.x;
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

	if (params.enableRestir)
	{
		Reservoir<PrefixPath> prefixRes{};
		PrefixGen(prefixRes, pixelCoord, origin, dir, rng);
		PrefixTempReuse(prefixRes);
	}
	else
	{
		outputRadiance = TraceCompletePath(origin, dir, 8, 8, rng, params);
	}

	// Store radiance output
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
