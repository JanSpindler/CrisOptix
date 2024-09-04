#pragma once

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
