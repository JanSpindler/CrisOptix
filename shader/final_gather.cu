#include <cuda_runtime.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>
#include <graph/trace.h>
#include <graph/restir/PrefixSearchPayload.h>

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

static __forceinline__ __device__ glm::vec3 GetPathContribution(const PrefixPath& prefix, const SuffixPath& suffix)
{
	if (prefix.valid)
	{
		if (prefix.nee)
		{
			return prefix.f / prefix.p;
		}
		else if (suffix.valid)
		{
			return (prefix.f * suffix.f) / (prefix.p * suffix.p);
		}
	}

	return glm::vec3(0.0f);
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

	// Get prefix and suffix from this pixels restir
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;
	const SuffixPath& suffix = params.restir.suffixReservoirs[pixelIdx].sample;

	// Init empty output radiance
	//glm::vec3 outputRadiance(0.0f);
	glm::vec3 outputRadiance = GetPathContribution(prefix, suffix);

	// Final gather
	// K = M - 1
	const size_t k = params.restir.gatherM - 1;
	if (k > 0)
	{
		for (size_t prefixIdx = 1; prefixIdx < params.restir.gatherN; ++prefixIdx)
		{
			// Trace new prefix for pixel q
			const PrefixPath neighPrefix{};

			// Find k neighboring prefixes in world space
			static constexpr float EPSILON = 1e-16;
			PrefixSearchPayload prefixSearchPayload(pixelIdx);
			TraceWithDataPointer<PrefixSearchPayload>(
				params.restir.prefixEntriesTraversHandle,
				prefix.lastInteraction.pos,
				glm::vec3(EPSILON),
				0.0f,
				EPSILON,
				params.restir.prefixEntriesTraceParams,
				&prefixSearchPayload);

			// TODO: Store in array
			const Reservoir<SuffixPath> neighSuffixRes[1] = { {} };

			// Borrow their suffixes and gather path contributions
			for (size_t suffixIdx = 0; suffixIdx < prefixSearchPayload.neighCount; ++suffixIdx)
			{
				// Calc mis weight m_i(Y_ij^S, X_i^P)
				const float misWeight = 1.0f;

				// Calc path contribution
				const glm::vec3 pathContrib = GetPathContribution(neighPrefix, neighSuffixRes[0].sample);

				// Calc ucw
				const float ucwPrefix = 1.0f / neighPrefix.p;
				const float ucwSuffix = 1.0f; // TODO: Shift and use ucw after shift
				const float ucw = ucwPrefix * ucwSuffix;

				// Gather
				//outputRadiance += misWeight * pathContrib * ucw;
			}
		}
	}

	// Accum
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
