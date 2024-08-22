#pragma once

#include <glm/glm.hpp>
#include <optix.h>
#include <graph/DeviceBuffer.h>
#include <graph/CuBufferView.h>

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
