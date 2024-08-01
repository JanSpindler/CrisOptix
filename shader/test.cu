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
#include <model/Emitter.h>

static constexpr uint32_t MAX_TRACE_OPS = 16;
static constexpr uint32_t MAX_TRACE_DEPTH = 0;

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

static constexpr __device__ EmitterSample SampleLightDir(const glm::vec3& currentPos, PCG32& rng)
{
	// Sample emitter
	const size_t emitterCount = params.emitterTable.count;
	const size_t emitterIdx = rng.NextUint64() % emitterCount;
	const EmitterData& emitter = params.emitterTable[emitterIdx];

	// Sample emitter point
	return emitter.SamplePoint(rng);
}

static constexpr __device__ float GetPHatDi(const SurfaceInteraction& interaction, const EmitterSample& emitterSample, PCG32& rng)
{
	const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
		interaction.meshSbtData->evalMaterialSbtIdx,
		interaction,
		glm::normalize(emitterSample.pos - interaction.pos));
	const float pHat = glm::length(brdfEvalResult.brdfResult);
	return pHat;
}

static constexpr __device__ Reservoir<EmitterSample>& GetDiReservoir(const uint32_t x, const uint32_t y)
{
	return params.diReservoirs[y * params.width + x];
}

static constexpr __device__ Reservoir<EmitterSample> CombineReservoirDi(
	const Reservoir<EmitterSample>& r1, 
	const Reservoir<EmitterSample>& r2,
	const SurfaceInteraction& interaction,
	PCG32& rng)
{
	const float pHat1 = GetPHatDi(interaction, r1.y, rng);
	const float pHat2 = GetPHatDi(interaction, r2.y, rng);

	Reservoir<EmitterSample> res = { {}, 0.0f, 0 };
	res.Update(r1.y, pHat1 * r1.W * r1.M, rng);
	res.Update(r2.y, pHat2 * r2.W * r2.M, rng);
	
	res.M = r1.M + r2.M;
	res.W = GetPHatDi(interaction, res.y, rng) * res.wSum / static_cast<float>(res.M);
}

static constexpr __device__ Reservoir<EmitterSample> RestirRis(const SurfaceInteraction& interaction, const size_t sampleCount, PCG32& rng)
{
	Reservoir<EmitterSample> reservoir = { {}, 0.0f, 0 };

	for (size_t idx = 0; idx < sampleCount; ++idx)
	{
		const EmitterSample emitterSample = SampleLightDir(interaction.pos, rng);
		const float pHat = GetPHatDi(interaction, emitterSample, rng);
		reservoir.Update(emitterSample, pHat / emitterSample.p, rng);
	}

	return reservoir;
}

static constexpr __device__ void RestirDi(
	const glm::uvec3& launchIdx,
	const SurfaceInteraction& interaction, 
	PCG32& rng)
{
	// Generate new samples
	Reservoir<EmitterSample> newReservoir = RestirRis(interaction, 4, rng);

	// Check if shadowed
	const bool occluded = TraceOcclusion(
		params.traversableHandle,
		interaction.pos,
		glm::normalize(newReservoir.y.pos - interaction.pos),
		1e-3f,
		glm::length(newReservoir.y.pos - interaction.pos),
		params.occlusionTraceParams);
	if (occluded) { newReservoir.W = 0.0f; }

	// Temporal reuse
	if (params.frameIdx > 1)
	{
		const glm::vec2 oldUV = params.cameraData.prevW2V * glm::vec4(interaction.pos, 1.0f);
		if (oldUV.x == glm::clamp(oldUV.x, 0.0f, 1.0f) && oldUV.y == glm::clamp(oldUV.y, 0.0f, 1.0f))
		{
			newReservoir = CombineReservoirDi(newReservoir, GetDiReservoir(launchIdx.x, launchIdx.y), interaction, rng);
		}
	}

	// Spatial reuse
	const size_t N = 2;
	for (size_t n = 0; n < 3; ++n)
	{
		const size_t nX = launchIdx.x + (rng.NextUint32() % (2 * N + 1)) - N;
		const size_t nY = launchIdx.y + (rng.NextUint32() % (2 * N + 1)) - N;
		if (nX < params.width && nY < params.height)
		{
			newReservoir = CombineReservoirDi(newReservoir, GetDiReservoir(nX, nY), interaction, rng);
		}
	}

	// Store reservoir
	GetDiReservoir(launchIdx.x, launchIdx.y) = newReservoir;
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
		nextRay.throughput = glm::vec3(1.0f);
		nextRay.depth = 0;
	}

	// Trace
	for (uint32_t traceIdx = 0; traceIdx < MAX_TRACE_OPS; ++traceIdx)
	{
		if (!nextRayValid) { break; }

		//
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

		// Decide if NEE or continue PT
		const float neeProb = 0.25f;
		if (rng.NextFloat() < neeProb || currentRay.depth >= MAX_TRACE_DEPTH)
		{
			// NEE
			// Sample light source
			RestirDi(launchIdx, interaction, rng);
			//const EmitterSample emitterSample = SampleLightDir(interaction.pos, rng);
			const EmitterSample emitterSample = GetDiReservoir(launchIdx.x, launchIdx.y).y;
			const glm::vec3 lightDir = glm::normalize(emitterSample.pos - interaction.pos);
			const float distance = glm::length(emitterSample.pos - interaction.pos);

			// Cast shadow ray
			const bool occluded = TraceOcclusion(
				params.traversableHandle,
				interaction.pos,
				lightDir,
				1e-3,
				distance,
				params.occlusionTraceParams);
			if (occluded) { continue; }

			// Calc brdf
			const BrdfEvalResult brdfEvalResult = optixDirectCall<BrdfEvalResult, const SurfaceInteraction&, const glm::vec3&>(
				interaction.meshSbtData->evalMaterialSbtIdx,
				interaction,
				lightDir);
			outputRadiance = currentRay.throughput * brdfEvalResult.brdfResult * emitterSample.color;
			if (currentRay.depth == 0) { outputRadiance += brdfEvalResult.emission; }

			// Exit from PT
			break;
		}

		// Indirect illumination, generate next ray
		BrdfSampleResult brdfSampleResult = optixDirectCall<BrdfSampleResult, const SurfaceInteraction&, PCG32&>(
			interaction.meshSbtData->sampleMaterialSbtIdx,
			interaction, 
			rng);
		if (brdfSampleResult.samplingPdf > 0.0f)
		{
			nextRay.origin = interaction.pos;
			nextRay.dir = brdfSampleResult.outDir;
			nextRay.throughput = currentRay.throughput * brdfSampleResult.weight;
			nextRay.depth = currentRay.depth + 1;
			nextRayValid = true;
		}
	}

	// Store radiance output
	const glm::vec3 oldVal = params.outputBuffer[pixelIdx];
	const float blendFactor = 1.0f / static_cast<float>(params.frameIdx + 1);
	//params.outputBuffer[pixelIdx] = blendFactor * outputRadiance + (1.0f - blendFactor) * oldVal;
	params.outputBuffer[pixelIdx] = outputRadiance;
}
