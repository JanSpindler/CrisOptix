#include <optix.h>
#include <cuda_runtime.h>
#include <util/glm_cuda.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/random.h>
#include <util/shader_util.h>
#include <graph/Interaction.h>
#include <graph/trace.h>

static constexpr uint32_t MAX_TRACE_OPS = 128;
static constexpr uint32_t MAX_TRACE_DEPTH = 8;

__constant__ LaunchParams params;

struct Ray
{
	glm::vec3 origin;
	glm::vec3 dir;
	glm::vec3 throughput; // Contribution to final radiance value of pixel
	uint32_t depth;
};

extern "C" __global__ void __miss__main()
{
	SurfaceInteraction* si = GetPayloadDataPointer<SurfaceInteraction>();

	const glm::vec3 world_ray_origin = cuda2glm(optixGetWorldRayOrigin());
	const glm::vec3 world_ray_dir = cuda2glm(optixGetWorldRayDirection());
	const float tmax = optixGetRayTmax();

	si->valid = false;
	si->inRayDir = world_ray_dir;
	si->inRayDist = tmax;
}

extern "C" __global__ void __raygen_main()
{
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());

	if (launchIdx.x >= params.width || launchIdx.y >= params.height || launchIdx.z >= 1)
	{
		return;
	}

	const uint32_t pixelIdx = launchIdx.y * launchDims.x + launchIdx.x;
	const uint64_t seed = SampleTEA64(pixelIdx, 1);
	PCG32 rng(seed);

	glm::vec3 outputRadiance(0.0f);

	bool nextRayValid = true;
	Ray nextRay{};
	{
		glm::vec2 uv = (glm::vec2(launchIdx) + rng.Next2d()) / glm::vec2(params.width, params.height);
		uv = 2.0f * uv - 1.0f; // [0, 1] -> [-1, 1]
		SpawnCameraRay(params.cameraData, uv, nextRay.origin, nextRay.dir);
		nextRay.throughput = glm::vec3(1);
		nextRay.depth = 0;
	}

	for (uint32_t traceIdx = 0; traceIdx < MAX_TRACE_OPS; ++traceIdx)
	{
		if (!nextRayValid) { break; }

		Ray currentRay = nextRay;
		nextRayValid = false;

		SurfaceInteraction interaction{};
		TraceWithDataPointer(
			params.traversableHandle, 
			currentRay.origin, 
			currentRay.dir, 
			1e-3, 
			1e16, 
			params.surfaceTraceParams, 
			&interaction);

		if (!interaction.valid) { continue; }

		// TODO: Emitter
		outputRadiance = glm::vec3(1.0f);

		if (currentRay.depth >= MAX_TRACE_DEPTH) { continue; }
	}

	params.outputBuffer[pixelIdx] = outputRadiance;
}
