#include <cuda_runtime.h>
#include <util/glm_cuda.h>
#include <optix_device.h>
#include <graph/Interaction.h>
#include <graph/trace.h>

static constexpr __forceinline__ __device__ glm::vec3 PointObjectToWorld(const glm::vec3& point)
{
	return cuda2glm(optixTransformPointFromObjectToWorldSpace(glm2cuda(point)));
}

static constexpr __forceinline__ __device__ glm::vec3 NormalObjectToWorld(const glm::vec3& normal)
{
	return glm::normalize(cuda2glm(optixTransformNormalFromObjectToWorldSpace(glm2cuda(normal))));
}

extern "C" __global__ void __closesthit__mesh()
{
	// Get interaction by ptr
	Interaction* si = GetPayloadDataPointer<Interaction>();
	const MeshSbtData* sbtData = *reinterpret_cast<const MeshSbtData**>(optixGetSbtDataPointer());
	si->meshSbtData = sbtData;

	// Fill ray info
	const glm::vec3 worldRayOrigin = cuda2glm(optixGetWorldRayOrigin());
	const glm::vec3 worldRayDir = cuda2glm(optixGetWorldRayDirection());
	
	// Fill basic interaction info
	si->inRayDir = worldRayDir;
	si->valid = true;

	// Get primitive data
	const uint32_t primIdx = optixGetPrimitiveIndex();
	const glm::vec2 baryCoord = cuda2glm(optixGetTriangleBarycentrics());
	si->instanceId = optixGetInstanceId();
	si->primitiveIdx = primIdx;

	// Indices of triangle vertices in the mesh
	const glm::uvec3 indices(
		sbtData->indices[primIdx * 3 + 0],
		sbtData->indices[primIdx * 3 + 1],
		sbtData->indices[primIdx * 3 + 2]);

	// Vertices
	const Vertex& v0 = sbtData->vertices[indices.x];
	const Vertex& v1 = sbtData->vertices[indices.y];
	const Vertex& v2 = sbtData->vertices[indices.z];

	// Interpolate
	si->pos = InterpolateBary<glm::vec3>(baryCoord, v0.pos, v1.pos, v2.pos);
	si->pos = PointObjectToWorld(si->pos);

	si->normal = InterpolateBary<glm::vec3>(baryCoord, v0.normal, v1.normal, v2.normal);
	si->normal = NormalObjectToWorld(si->normal);

	si->tangent = InterpolateBary<glm::vec3>(baryCoord, v0.tangent, v1.tangent, v2.tangent);
	si->tangent = NormalObjectToWorld(si->tangent);

	si->uv = InterpolateBary<glm::vec2>(baryCoord, v0.uv, v1.uv, v2.uv);
}
