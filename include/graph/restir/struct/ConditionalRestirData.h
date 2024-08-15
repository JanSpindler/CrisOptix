#pragma once

#include <glm/glm.hpp>
#include <graph/restir/struct/PathReconInfo.h>
#include <graph/restir/struct/PathRisState.h>

struct ConditionalRestirData
{
    PathRisState pathRis;
    PathReconInfo pathReconInfo;
    float lastScatterPdf;
    glm::vec3 lastScatterWeight;
    float rrPdf;

    __forceinline__ __device__ __host__ ConditionalRestirData() :
        pathRis(),
        pathReconInfo(),
        lastScatterPdf(0.0f),
        lastScatterWeight(0.0f),
        rrPdf(1.0f)
    {
    }
};
