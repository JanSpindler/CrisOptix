#pragma once

#include <glm/glm.hpp>
#include <util/f16.h>

//#define USE_CUDA_F16

#ifdef USE_CUDA_F16

typedef F16Vec2 Vec2;
typedef F16Vec3 Vec3;

#else

typedef glm::vec2 Vec2;
typedef glm::vec3 Vec3;

#endif
