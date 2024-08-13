#pragma once

#include <graph/Interaction.h>

struct PrefixGBuffer
{
	SurfaceInteraction interaction;
	glm::vec3 outDir;

	constexpr __device__ PrefixGBuffer() :
		interaction({}),
		outDir({ 0.0f, 0.0f, 0.0f })
	{
	}

	constexpr __device__ PrefixGBuffer(const SurfaceInteraction& _interaction, const glm::vec3& _outDir) :
		interaction(_interaction),
		outDir(_outDir)
	{
	}
};
