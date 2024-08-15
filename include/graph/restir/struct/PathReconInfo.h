#pragma once

#include <graph/restir/struct/RestirPathFlags.h>
#include <util/random.h>

struct PathReconInfo
{
    // also stores random replay path information
    RestirPathFlags pathFlags;
    PCG32 reconRng;         // to recover NEE information
    PCG32 initRng;
    PCG32 suffixInitRng;
    // also stores outgoing direction at the previous vertex of rcVertex, used in hybrid shift replay
    glm::vec3 reconThroughput; // path throughput after rcVertex
    // also stores the hit for the previous vertex of rcVertex, used in hybrid shift replay

    SurfaceInteraction reconHit;

    float reconJacobian;
    glm::vec3 reconInDir; // can be computed outside
};
