#pragma once

#include <graph/Interaction.h>

struct ReconData
{
	SurfaceInteraction pos0Interaction;
	glm::vec3 pos1;
	glm::vec3 outDir;
	glm::vec3 throughput;

	__forceinline__ __device__ __host__ ReconData() :
		pos0Interaction({}),
		pos1(0.0f),
		outDir(0.0f),
		throughput(0.0f)
	{
	}

	__forceinline__ __device__ __host__ ReconData(
		const SurfaceInteraction& _pos0Interaction, 
		const glm::vec3& _pos1, 
		const glm::vec3& _outDir, 
		const glm::vec3& _throughput) 
		:
		pos0Interaction(_pos0Interaction),
		pos1(_pos1),
		outDir(_outDir),
		throughput(_throughput)
	{
	}
};
