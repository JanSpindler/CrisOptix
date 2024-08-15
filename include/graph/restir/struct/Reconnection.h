#pragma once

#include <graph/brdf.h>
#include <cuda_runtime.h>

struct Reconnection
{
	BrdfSampleResult pos0Brdf;
	BrdfSampleResult pos1Brdf;

	__forceinline__ __device__ glm::vec3 GetThroughput() const
	{
		return pos0Brdf.weight * pos1Brdf.weight / GetP();
	}

	__forceinline__ __device__ float GetP() const
	{
		return pos0Brdf.samplingPdf * pos1Brdf.samplingPdf;
	}

	__forceinline__ __device__ float GetWeight() const
	{
		return GetLuminance(GetThroughput()) / GetP();
	}
};
