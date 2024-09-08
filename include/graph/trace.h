#pragma once

#include <cuda_runtime.h>
#include <graph/TraceParameters.h>
#include <optix.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/Interaction.h>
#include <graph/LaunchParams.h>

static constexpr __forceinline__ __host__ __device__ void PackPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static constexpr __forceinline__ __host__ __device__ void* UnpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

template <typename T>
static constexpr __forceinline__ __device__ void TraceWithDataPointer(
    OptixTraversableHandle handle,
    const glm::vec3&              ray_origin,
    const glm::vec3&              ray_direction,
    const float                  tmin,
    const float                  tmax,
    const TraceParameters& trace_params,
    T& payload_ptr
)
{
    uint32_t u0, u1;
    PackPointer(&payload_ptr, u0, u1);
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

template <typename T>
static constexpr __forceinline__ __device__ T* GetPayloadDataPointer()
{
    // Get the pointer to the payload data
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(UnpackPointer(u0, u1));
}

static constexpr __forceinline__ __device__ bool TraceOcclusion(
    const glm::vec3&              ray_origin,
    const glm::vec3&              ray_direction,
    const float                  tmin,
    const float                  tmax,
    const LaunchParams& params)
{
    const TraceParameters& trace_params = params.occlusionTraceParams;

    uint32_t occluded = 1u;
    optixTrace(
        params.traversableHandle,
        glm2cuda(ray_origin),
        glm2cuda(ray_direction),
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),  // visibilityMask
        trace_params.rayFlags,   // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        trace_params.sbtOffset,
        trace_params.sbtStride,
        trace_params.missSbtIdx,
        occluded);              // payload 0
    return occluded != 0;
}

static constexpr __forceinline__ __device__ void SetOcclusionPayload(bool occluded)
{
    // Set the payload that _this_ ray will yield
    optixSetPayload_0(static_cast<uint32_t>(occluded));
}

//static constexpr __forceinline__ __device__ void TraceInteractionSeed(
//    const InteractionSeed& seed, 
//    Interaction& interaction, 
//    const LaunchParams& params)
//{
//    TraceWithDataPointer<Interaction>(
//        params.traversableHandle,
//        seed.pos - static_cast<glm::vec3>(seed.inDir),
//        seed.inDir,
//        1e-3f,
//        1.0f + 1e-3,
//        params.surfaceTraceParams,
//        interaction);
//}
