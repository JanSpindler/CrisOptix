#pragma once

#include <glm/glm.hpp>
#include <model/Mesh.h>

struct SurfaceInteraction
{
	bool valid;
	glm::vec3 inRayDir;
	float inRayDist;
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	uint32_t primitiveIdx;
	const MeshSbtData* meshSbtData;
};
