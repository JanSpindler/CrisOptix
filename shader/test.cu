#include <optix.h>
#include <cuda_runtime.h>
#include <util/glm_cuda.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/random.h>
#include <util/shader_util.h>
#include <graph/Interaction.h>
#include <graph/trace.h>
#include <graph/brdf.h>

static constexpr uint32_t MAX_TRACE_OPS = 32;
static constexpr uint32_t MAX_TRACE_DEPTH = 8;

__constant__ LaunchParams params;

struct Ray
{
	glm::vec3 origin;
	glm::vec3 dir;
	glm::vec3 throughput; // Contribution to final radiance value of pixel
	uint32_t depth;
};

extern "C" __global__ void __closesthit__mesh()
{
	// Get interaction by ptr
	SurfaceInteraction* si = GetPayloadDataPointer<SurfaceInteraction>();
	const MeshSbtData* sbtData = reinterpret_cast<const MeshSbtData*>(optixGetSbtDataPointer());
	si->meshSbtData = sbtData;

	// Fill ray info
	const glm::vec3 worldRayOrigin = cuda2glm(optixGetWorldRayOrigin());
	const glm::vec3 worldRayDir = cuda2glm(optixGetWorldRayDirection());
	const float tMax = optixGetRayTmax();

	// Fill basic interaction info
	si->inRayDir = worldRayDir;
	si->inRayDist = tMax;
	si->valid = true;

	// Get primitive data
	const uint32_t primIdx = optixGetPrimitiveIndex();
	const glm::vec2 baryCoord = cuda2glm(optixGetTriangleBarycentrics());

	// Indices of triangle vertices in the mesh
	glm::uvec3 tri = glm::uvec3(0u);

	// Indices stored as 32-bit unsigned integers
	//const glm::u32vec3* indices = reinterpret_cast<glm::u32vec3*>(mesh_data->indices.data);
	//tri = glm::uvec3(indices[primIdx]);
}

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

extern "C" __global__ void __miss__occlusion()
{
	SetOcclusionPayload(false);
}

extern "C" __global__ void __raygen__main()
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

	//params.outputBuffer[pixelIdx] = glm::vec3(0.5f);
	//return;

	for (uint32_t traceIdx = 0; traceIdx < MAX_TRACE_OPS; ++traceIdx)
	{
		if (!nextRayValid) { break; }

		Ray currentRay = nextRay;
		nextRayValid = false;

		SurfaceInteraction interaction{};
		TraceWithDataPointer<SurfaceInteraction>(
			params.traversableHandle, 
			currentRay.origin, 
			currentRay.dir, 
			1e-3, 
			1e16, 
			params.surfaceTraceParams, 
			&interaction);

		if (!interaction.valid) { continue; }

		const glm::vec3 dirLightDir(0.0f, 0.0f, 1.0f);
		const BrdfResult brdfResult = optixDirectCall<BrdfResult, const SurfaceInteraction&, const glm::vec3&>(
			interaction.meshSbtData->evalMaterialSbtIdx, 
			interaction, 
			dirLightDir);
		outputRadiance = brdfResult.brdfResult;

		if (currentRay.depth >= MAX_TRACE_DEPTH) { continue; }
	}

	params.outputBuffer[pixelIdx] = outputRadiance;
}
