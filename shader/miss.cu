#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <graph/Interaction.h>
#include <graph/trace.h>
#include <graph/restir/PrefixSearchPayload.h>

extern "C" __global__ void __miss__prefix_entry()
{
}

extern "C" __global__ void __miss__main()
{
	Interaction* si = GetPayloadDataPointer<Interaction>();

	const glm::vec3 world_ray_origin = cuda2glm(optixGetWorldRayOrigin());
	const glm::vec3 world_ray_dir = cuda2glm(optixGetWorldRayDirection());

	si->valid = false;
	si->inRayDir = world_ray_dir;
}

extern "C" __global__ void __miss__occlusion()
{
	SetOcclusionPayload(false);
}
