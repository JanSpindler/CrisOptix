#pragma once

#include <glm/glm.hpp>

struct Interaction
{
	bool valid;
	glm::vec3 inRayDir;
	float inRayDist;
	glm::vec3 pos;
};

struct SurfaceInteraction : Interaction
{
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	uint32_t primitiveIdx;
};
