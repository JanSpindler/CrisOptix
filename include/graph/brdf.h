#pragma once

#include <glm/glm.hpp>

struct BrdfEvalResult
{
    // BSDF value for the given direction. NOTE: This is not divided by the BSDF sampling PDF!
    glm::vec3 brdfResult;
    
    // The probability of sampling the given direction via BSDF importance sampling.
    // The sampling PDF is useful for multiple importance sampling.
    float samplingPdf;

    // Technically not in brdf
    glm::vec3 emission;
};

struct BrdfSampleResult
{
    // The sampled outgoing ray direction.
    glm::vec3 outDir;
    // BSDF value divided by sampling PDF for the sampled direction.
    glm::vec3 weight;
    // The sampling PDF is useful for multiple importance sampling.
    float samplingPdf;
};
