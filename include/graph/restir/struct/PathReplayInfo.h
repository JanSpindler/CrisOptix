#pragma once

#include <graph/restir/struct/RestirPathFlags.h>
#include <glm/glm.hpp>
#include <graph/Interaction.h>
#include <util/random.h>

struct PathReplayInfo
{
    RestirPathFlags pathFlags;
    glm::vec3 reconThroughput; // path throughput after reconnection vertex

    SurfaceInteraction reconHit;

    float reconJacobian;
    PCG32 suffixInitRng;

    glm::vec3 inDir;
};
