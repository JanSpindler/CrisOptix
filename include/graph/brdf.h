#pragma once

#include <glm/glm.hpp>

struct BrdfResult
{
    // BSDF value for the given direction. NOTE: This is not divided by the BSDF sampling PDF!
    glm::vec3 brdfResult;
    
    // The probability of sampling the given direction via BSDF importance sampling.
    // The sampling PDF is useful for multiple importance sampling.
    float samplingPdf;

    // Technically not in brdf
    glm::vec3 emission;
};
