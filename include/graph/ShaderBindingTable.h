#pragma once

#include <optix.h>
#include <vector>
#include <graph/DeviceBuffer.h>
#include <tuple>

class ShaderBindingTable
{
public:
	ShaderBindingTable(
		const OptixDeviceContext optixDeviceContext, 
		const std::vector<OptixProgramGroup>& raygenProgramGroups,
		const std::vector<OptixProgramGroup>& missProgramGroups,
		const std::vector<OptixProgramGroup>& exceptionProgramGroups,
		const std::vector<OptixProgramGroup>& callableProgramGroups,
		const std::vector<OptixProgramGroup>& hitgroupProgramGroups);
	~ShaderBindingTable();

	const OptixShaderBindingTable* GetSbt(const size_t raygenIdx) const;

private:
	std::vector<OptixShaderBindingTable> m_Sbts{};
	DeviceBuffer<char> m_Buffer{};

	static void AddRecords(
		const std::vector<OptixProgramGroup>& programGroups,
		const size_t stride,
		std::vector<char>& recordsData);

	std::tuple<CUdeviceptr, uint32_t, uint32_t> GetBaseStrideCount(
		const std::vector<OptixProgramGroup>& programGroups,
		const size_t offset,
		const size_t stride) const;
};
