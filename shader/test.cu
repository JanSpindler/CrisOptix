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
#include <util/random.h>

static constexpr uint32_t MAX_TRACE_DEPTH = 8;

__constant__ LaunchParams params;

struct Ray
{
	glm::vec3 origin;
	glm::vec3 dir;
	glm::vec3 throughput; // Contribution to final radiance value of pixel
	uint32_t depth;
};

static constexpr __device__ glm::vec3 PointObjectToWorld(const glm::vec3& point)
{
	return cuda2glm(optixTransformPointFromObjectToWorldSpace(glm2cuda(point)));
}

static constexpr __device__ glm::vec3 NormalObjectToWorld(const glm::vec3& normal)
{
	return glm::normalize(cuda2glm(optixTransformNormalFromObjectToWorldSpace(glm2cuda(normal))));
}

template <typename T>
static constexpr __device__ T InterpolateBary(const glm::vec2& baryCoord, const T& v0, const T& v1, const T& v2)
{
	return 
		(1.0f - baryCoord.x - baryCoord.y) * v0 
		+ baryCoord.x * v1 
		+ baryCoord.y * v2;
}

extern "C" __global__ void __closesthit__mesh()
{
	// Get interaction by ptr
	SurfaceInteraction* si = GetPayloadDataPointer<SurfaceInteraction>();
	const MeshSbtData* sbtData = *reinterpret_cast<const MeshSbtData**>(optixGetSbtDataPointer());
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
	si->primitiveIdx = primIdx;

	// Indices of triangle vertices in the mesh
	const glm::uvec3 indices(
		sbtData->indices[primIdx * 3 + 0], 
		sbtData->indices[primIdx * 3 + 1], 
		sbtData->indices[primIdx * 3 + 2]);

	// Vertices
	const Vertex v0 = sbtData->vertices[indices.x];
	const Vertex v1 = sbtData->vertices[indices.y];
	const Vertex v2 = sbtData->vertices[indices.z];

	// Interpolate
	si->pos = InterpolateBary<glm::vec3>(baryCoord, v0.pos, v1.pos, v2.pos);
	si->pos = PointObjectToWorld(si->pos);

	si->normal = InterpolateBary<glm::vec3>(baryCoord, v0.normal, v1.normal, v2.normal);
	si->normal = NormalObjectToWorld(si->normal);

	si->tangent = InterpolateBary<glm::vec3>(baryCoord, v0.tangent, v1.tangent, v2.tangent);
	si->tangent = NormalObjectToWorld(si->tangent);

	si->uv = InterpolateBary<glm::vec2>(baryCoord, v0.uv, v1.uv, v2.uv);
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

struct EmitterSample
{
	glm::vec3 dir;
	glm::vec3 color;
	float distance;
};

static constexpr __device__ EmitterSample SampleLightDir(const glm::vec3 currentPos, PCG32& rng)
{
	const size_t emitterCount = params.emitterTable.count;
	const size_t emitterIdx = rng.NextUint64() % emitterCount;
	const EmitterData& emitter = params.emitterTable[emitterIdx];
	const size_t faceIdx = rng.NextUint64() % emitter.areaBuffer.count; // TODO: Use area or solid angle sampling

	const glm::vec3 v0 = emitter.vertexBuffer[emitter.indexBuffer[faceIdx * 3 + 0]].pos;
	const glm::vec3 v1 = emitter.vertexBuffer[emitter.indexBuffer[faceIdx * 3 + 1]].pos;
	const glm::vec3 v2 = emitter.vertexBuffer[emitter.indexBuffer[faceIdx * 3 + 2]].pos;

	const float r1 = rng.NextFloat();
	const float r2 = rng.NextFloat();
	const glm::vec3 emitterPoint = (1.0f - r1 - r2) * v0 + r1 * v1 + r2 * v2;

	const glm::vec3 lightDir = glm::normalize(emitterPoint - currentPos);
	const float distance = glm::length(emitterPoint - currentPos);
	
	return { lightDir, emitter.color, distance };
}

extern "C" __global__ void __raygen__main()
{
	//
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());

	// Exit if invalid launch idx
	if (launchIdx.x >= params.width || launchIdx.y >= params.height || launchIdx.z >= 1)
	{
		return;
	}

	// Init RNG
	const uint32_t pixelIdx = launchIdx.y * launchDims.x + launchIdx.x;
	const uint64_t seed = SampleTEA64(pixelIdx, params.frameIdx);
	PCG32 rng(seed);

	// Init radiance with 0
	glm::vec3 outputRadiance(0.0f);

	// Spawn camera ray
	bool nextRayValid = true;
	Ray nextRay{};
	{
		glm::vec2 uv = (glm::vec2(launchIdx) + rng.Next2d()) / glm::vec2(params.width, params.height);
		uv = 2.0f * uv - 1.0f; // [0, 1] -> [-1, 1]
		SpawnCameraRay(params.cameraData, uv, nextRay.origin, nextRay.dir);
		nextRay.throughput = glm::vec3(1);
		nextRay.depth = 0;
	}

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < MAX_TRACE_DEPTH; ++traceIdx)
	{
		if (!nextRayValid) { break; }

		Ray currentRay = nextRay;
		nextRayValid = false;

		// Sample surface interaction
		SurfaceInteraction interaction{};
		TraceWithDataPointer<SurfaceInteraction>(
			params.traversableHandle, 
			currentRay.origin, 
			currentRay.dir, 
			1e-3, 
			1e16, 
			params.surfaceTraceParams, 
			&interaction);

		// Exit if no surface found
		if (!interaction.valid) { continue; }

		// Sample light source
		const EmitterSample emitterSample = SampleLightDir(interaction.pos, rng);

		// Cast shadow ray
		const bool occluded = TraceOcclusion(
			params.traversableHandle,
			interaction.pos,
			emitterSample.dir,
			1e-3,
			emitterSample.distance,
			params.occlusionTraceParams);
		if (occluded) { continue; }

		// Calc brdf
		const BrdfResult brdfResult = optixDirectCall<BrdfResult, const SurfaceInteraction&, const glm::vec3&>(
			interaction.meshSbtData->evalMaterialSbtIdx, 
			interaction, 
			emitterSample.dir);
		outputRadiance = brdfResult.brdfResult * emitterSample.color;
		if (traceIdx == 0) { outputRadiance += brdfResult.emission; }
	}

	// Store radiance output
	params.outputBuffer[pixelIdx] = outputRadiance;
}
