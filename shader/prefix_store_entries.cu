#include <cuda_runtime.h>
#include <optix_device.h>
#include <graph/LaunchParams.h>
#include <util/glm_cuda.h>
#include <util/pixel_index.h>

__constant__ LaunchParams params;

extern "C" __global__ void __raygen__prefix_store_entries()
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

	// Get prefix and suffix
	const PrefixPath& prefix = params.restir.prefixReservoirs[pixelIdx].sample;
	const SuffixPath& suffix = params.restir.suffixReservoirs[pixelIdx].sample;

	// Store prefix entry aabb
	const float radius = params.restir.gatherRadius;
	const glm::vec3& pos = prefix.lastInteraction.pos;

	OptixAabb& aabb = params.restir.prefixEntryAabbs[pixelIdx];
	if (prefix.valid && suffix.valid)
	{
		aabb.minX = pos.x - radius;
		aabb.minY = pos.y - radius;
		aabb.minZ = pos.z - radius;
		aabb.maxX = pos.x + radius;
		aabb.maxY = pos.y + radius;
		aabb.maxZ = pos.z + radius;
	}
	else
	{
		aabb = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	}
}
