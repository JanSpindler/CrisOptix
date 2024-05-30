#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <glm/glm.hpp>

struct TraceParameters
{
    // OptixVisibilityMask visibilityMask;
    // Combination of flags from `OptixRayFlags`.
    uint32_t            rayFlags;
    // Gobal offset to be applied to all accesses to hitgroup entries in the shader binding table (SBT).
    uint32_t            sbtOffset;
    // When an instance acceleration structure (IAS) contains multiple geometry acceleration structures (GAS), this stride is used to advance to the entry in the SBT corresponding to the respective GAS.
    uint32_t            sbtStride;
    // The index of the miss program to use when the ray does not intersect the scene geometry.
    uint32_t            missSbtIdx;
};

#ifdef __CUDACC__

#include <optix_device.h>

__host__ __device__ void PackPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
__forceinline__ __device__ void TraceWithDataPointer(
    OptixTraversableHandle handle,
    glm::vec3              ray_origin,
    glm::vec3              ray_direction,
    float                  tmin,
    float                  tmax,
    TraceParameters        trace_params,
    T* payload_ptr
)
{
    uint32_t u0, u1;
    PackPointer(payload_ptr, u0, u1);
    optixTrace(
        handle,
        glm2cuda(ray_origin),
        glm2cuda(ray_direction),
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),  // visibilityMask
        trace_params.rayFlags, // OPTIX_RAY_FLAG_NONE
        trace_params.sbtOffset,
        trace_params.sbtStride,
        trace_params.missSbtIdx,
        u0,                      // payload 0
        u1);                    // payload 1
    // optixTrace operation will have updated content of *payload_ptr
}

#endif
