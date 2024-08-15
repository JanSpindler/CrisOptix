#pragma once

#include <glm/glm.hpp>
#include <util/random.h>
#include <graph/Interaction.h>

struct PrefixPath
{
	bool valid;

	SurfaceInteraction lastInteraction;
	
	glm::vec3 throughput;
	float p;

	uint32_t len;
};
