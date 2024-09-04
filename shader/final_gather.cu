#include <cuda_runtime.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>
#include <graph/trace.h>
#include <graph/restir/PrefixSearchPayload.h>
#include <graph/restir/path_gen.h>
#include <cuda/std/tuple>

__constant__ LaunchParams params;

extern "C" __global__ void __intersection__prefix_entry()
{
	// Get pixel index of hit
	const uint32_t neighPixelIdx = optixGetPrimitiveIndex();

	// Get payload
	PrefixSearchPayload* payload = GetPayloadDataPointer<PrefixSearchPayload>();

	// Check if radius is truly as desired
	const glm::vec3 queryPos = cuda2glm(optixGetWorldRayOrigin());
	const glm::vec3& neighPos = params.restir.prefixReservoirs[neighPixelIdx].sample.lastInteraction.pos;
	const float distance = glm::distance(queryPos, neighPos);
	if (distance > params.restir.gatherRadius) { return; }

	// Build neighbor
	const PrefixNeighbor neigh(neighPixelIdx, distance);

	// Store neigh pixel idx
	const uint32_t k = params.restir.gatherM - 1;
	const uint32_t offset = payload->pixelIdx * k;

	// If neigh pixel idx buffer not full
	if (payload->neighCount < k)
	{
		// Append neigh pixel idx to buffer
		params.restir.prefixNeighbors[offset + payload->neighCount] = neigh;

		// Inc neigh count
		++payload->neighCount;

		// Find stored neigh with largest distance
		if (payload->neighCount == k)
		{
			payload->FindLargestDist(params);
		}
	}
	// If neigh pixel idx buffer is full AND the distance of new neigh is lower than the max distance so far
	else if (distance < payload->maxNeighDist)
	{
		params.restir.prefixNeighbors[offset + payload->maxDistNeighIdx] = neigh;
		payload->FindLargestDist(params);
	}
}

static __forceinline__ __device__ glm::vec3 GetPathContribution(
	const PrefixPath& prefix, 
	const SuffixPath& suffix,
	const float prefixUcw,
	const float suffixUcw)
{
	glm::vec3 output(0.0f);

	if (prefix.valid)
	{
		if (prefix.nee)
		{
			output = prefix.f * prefixUcw;;
		}
		else if (suffix.valid)
		{
			output = prefix.f * suffix.f * prefixUcw * suffixUcw;
		}
	}

	if (glm::any(glm::isinf(output) || glm::isnan(output))) { return glm::vec3(0.0f); }
	return output;
}

static __forceinline__ __device__ cuda::std::pair<glm::vec3, float> ShiftSuffix(const PrefixPath& prefix, const SuffixPath& suffix)
{
	// Trace occlusion
	const bool occluded = TraceOcclusion(
		params.traversableHandle,
		prefix.lastInteraction.pos,
		glm::normalize(suffix.reconInteraction.pos - prefix.lastInteraction.pos),
		1e-3f,
		glm::distance(suffix.reconInteraction.pos, prefix.lastInteraction.pos),
		params.occlusionTraceParams);
	if (occluded) { return { glm::vec3(0.0f), 0.0f }; }

	return { suffix.f, 1.0f / suffix.p };
}

static __forceinline__ __device__ glm::vec3 GetRadiance(const glm::uvec3& launchIdx, const size_t pixelIdx, PCG32& rng)
{
	// Get prefix and suffix from this pixels restir
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;
	const SuffixPath& suffix = params.restir.suffixReservoirs[pixelIdx].sample;

	// Exit if prefix is invalid
	if (!params.restir.restirGBuffers[pixelIdx].primaryInteraction.valid)
	{
		return glm::vec3(0.0f);
	}

	// Init empty output radiance
	glm::vec3 outputRadiance(0.0f);

	// Final gather
	// Spawn camera ray
	glm::vec3 origin(0.0f);
	glm::vec3 dir(0.0f);
	glm::vec2 uv = (glm::vec2(launchIdx) + rng.Next2d()) / glm::vec2(params.width, params.height);
	uv = 2.0f * uv - 1.0f; // [0, 1] -> [-1, 1]
	SpawnCameraRay(params.cameraData, uv, origin, dir);

	// K = M - 1
	float canonSuffixMisWeight = 1.0f;
	const size_t k = params.restir.gatherM - 1;
	if (k > 0)
	{
		for (size_t prefixIdx = 0; prefixIdx < params.restir.gatherN; ++prefixIdx)
		{
			// Trace new prefix for pixel q
			SurfaceInteraction interaction{};
			const PrefixPath neighPrefix = prefixIdx == 0 ? prefix : TracePrefix(origin, dir, params.restir.minPrefixLen, 8, interaction, rng, params);
			if (!neighPrefix.valid) { continue; }

			// Find k neighboring prefixes in world space
			static constexpr float EPSILON = 1e-16;
			PrefixSearchPayload prefixSearchPayload(pixelIdx);
			TraceWithDataPointer<PrefixSearchPayload>(
				params.restir.prefixEntriesTraversHandle,
				neighPrefix.lastInteraction.pos,
				glm::vec3(EPSILON),
				0.0f,
				EPSILON,
				params.restir.prefixEntriesTraceParams,
				&prefixSearchPayload);
			const uint32_t neighCount = prefixSearchPayload.neighCount;
			const float misWeight = 1.0f / static_cast<float>(neighCount + 1.0f);

			// Set mis weight for canonical suffix
			if (prefixIdx == 0)
			{
				canonSuffixMisWeight = misWeight;
			}

			// Track prefix stats
			if (params.restir.trackPrefixStats)
			{
				atomicMin(&params.restir.prefixStats[0].minNeighCount, neighCount);
				atomicMax(&params.restir.prefixStats[0].maxNeighCount, neighCount);
				atomicAdd(&params.restir.prefixStats[0].totalNeighCount, neighCount);
			}

			// Borrow their suffixes and gather path contributions
			for (size_t suffixIdx = 0; suffixIdx < neighCount; ++suffixIdx)
			{
				// Assume: Neighbor prefix and suffix are valid

				// Get suffix
				const uint32_t suffixPixelIdx = params.restir.prefixNeighbors[k * pixelIdx + suffixIdx].pixelIdx;
				const SuffixPath& neighSuffix = params.restir.suffixReservoirs[suffixPixelIdx].sample;

				// Shift suffix
				const cuda::std::pair<glm::vec3, float> shiftedSuffix = ShiftSuffix(neighPrefix, neighSuffix);
				const glm::vec3& shiftedF = shiftedSuffix.first;
				const float ucwSuffix = shiftedSuffix.second;

				// Calc path contribution
				const glm::vec3 pathContrib = glm::max(glm::vec3(0.0f), neighPrefix.f * shiftedF);

				// Calc ucw
				const float ucw = ucwSuffix / neighPrefix.p;

				// Gather
				outputRadiance += misWeight * pathContrib * ucw;
			}
		}

		outputRadiance /= static_cast<float>(params.restir.gatherN);
	}

	// Add canon suffix contrib
	outputRadiance += canonSuffixMisWeight * TraceCompletePath(origin, dir, 8, 8, rng, params);

	if (glm::any(glm::isnan(outputRadiance) || glm::isinf(outputRadiance))) { outputRadiance = glm::vec3(0.0f); }

	return outputRadiance;
}

extern "C" __global__ void __raygen__final_gather()
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
	PCG32& rng = params.restir.restirGBuffers[pixelIdx].rng;

	// Accum
	const glm::vec3 outputRadiance = GetRadiance(launchIdx, pixelIdx, rng);
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
