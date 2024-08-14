#pragma once

#include <graph/Interaction.h>
#include <cuda_runtime.h>

struct PrefixGBuffer
{
	SurfaceInteraction interaction;
	glm::vec3 outDir;

	__forceinline__ PrefixGBuffer() :
		interaction({}),
		outDir({ 0.0f, 0.0f, 0.0f })
	{
	}

	__forceinline__ PrefixGBuffer(const SurfaceInteraction& _interaction, const glm::vec3& _outDir) :
		interaction(_interaction),
		outDir(_outDir)
	{
	}
};
