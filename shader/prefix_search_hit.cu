#include <cuda_runtime.h>
#include <graph/LaunchParams.h>
#include <optix_device.h>
#include <graph/restir/PrefixSearchPayload.h>
#include <graph/trace.h>

__constant__ LaunchParams params;

extern "C" __global__ void __intersection__prefix_entry()
{
	// Get pixel index of hit
	const uint32_t neighPixelIdx = optixGetPrimitiveIndex();

	// Get payload
	PrefixSearchPayload* payload = GetPayloadDataPointer<PrefixSearchPayload>();

	// Get prefix
	const Reservoir<PrefixPath>& prefixRes = params.restir.prefixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
	const PrefixPath& prefix = prefixRes.sample;
	if (!prefix.IsValid()) { return; }

	// Get neighbor last interaction
	const Interaction neighLastInt(prefix.lastInt, params.transforms);
	if (!neighLastInt.valid) { return; }

	// Get suffix
	const Reservoir<SuffixPath>& suffixRes = params.restir.suffixReservoirs[2 * neighPixelIdx + params.restir.frontBufferIdx];
	const SuffixPath& suffix = suffixRes.sample;
	if (!suffix.IsValid()) { return; }

	// Get radius
	const OptixAabb& aabb = params.restir.prefixEntryAabbs[neighPixelIdx];
	const float radius = (aabb.maxX - aabb.minX) / 2.0f;

	// Check if radius is truly as desired
	const glm::vec3 queryPos = cuda2glm(optixGetWorldRayOrigin());
	const glm::vec3& neighPos = neighLastInt.pos;
	const float distance = glm::distance(queryPos, neighPos);
	if (distance > radius)
	{
		return;
	}
	++payload->intersectionCount;

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
