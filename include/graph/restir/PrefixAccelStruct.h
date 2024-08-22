#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <graph/DeviceBuffer.h>
#include <graph/CuBufferView.h>
#include <graph/LaunchParams.h>

struct PrefixEntryResult
{
	// Number of found neighbors.
	uint32_t neighCount;

	// Distance to furthest neighbor.
	float maxNeighDist;

	// Index of furthest neighbor.
	uint32_t maxDistNeighIdx;

	__forceinline__ __device__ __host__ PrefixEntryResult() :
		neighCount(0),
		maxNeighDist(0.0f),
		maxDistNeighIdx(0)
	{
	}

	__forceinline__ __device__ void FindLargestDist(const uint32_t pixelIdx, const LaunchParams& params)
	{
		if (neighCount == 0) { return; }

		const uint32_t k = params.restir.gatherM - 1;
		const uint32_t offset = pixelIdx * k;

		const glm::vec3& currPos = params.restir.prefixReservoirs[pixelIdx].sample.lastInteraction.pos;

		maxDistNeighIdx = 0;
		maxNeighDist = params.restir.prefixNeighPixels[offset + maxDistNeighIdx];

		for (uint32_t neighIdx = 0; neighIdx < k; ++neighIdx)
		{
			const uint32_t neighPixelIdx = params.restir.prefixNeighPixels[offset + neighIdx];
			const glm::vec3& neighPos = params.restir.prefixReservoirs[neighPixelIdx].sample.lastInteraction.pos;
			const float dist = glm::distance(currPos, neighPos);
			if (dist < maxNeighDist)
			{
				maxDistNeighIdx = neighIdx;
				maxNeighDist = dist;
			}
		}
	}
};

class PrefixAccelStruct
{
public:
	PrefixAccelStruct(const size_t size, const size_t neighCount, const OptixDeviceContext context);

	void Rebuild(const size_t neighCount);

	CuBufferView<OptixAabb> GetAabbBufferView() const;
	CuBufferView<uint32_t> GetNeighPixelBufferView() const;
	OptixTraversableHandle GetTlas() const;

	void SetSbtOffset(const uint32_t sbtOffset);

private:
	OptixDeviceContext m_Context = nullptr;

	OptixTraversableHandle m_GasHandle = 0;
	DeviceBuffer<uint8_t> m_GasBuf{};

	OptixTraversableHandle m_TlasHandle = 0;
	DeviceBuffer<uint8_t> m_TlasBuf{};

	DeviceBuffer<OptixAabb> m_Aabbs{};
	CUdeviceptr m_AabbDevPtr = 0;

	DeviceBuffer<uint32_t> m_NeighPixelBuf{};

	size_t m_PixelCount = 0;
	size_t m_NeighCount = 0;
	uint32_t m_SbtOffset = 0;

	void BuildGas();
	void BuildTlas();
};
