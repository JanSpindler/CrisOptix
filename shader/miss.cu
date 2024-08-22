#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <graph/Interaction.h>
#include <graph/trace.h>
#include <graph/restir/PrefixAccelStruct.h>

extern "C" __global__ void __miss__prefix_entry()
{
	PrefixEntryResult* result = GetPayloadDataPointer<PrefixEntryResult>();
	result->neighCount = 0;
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
