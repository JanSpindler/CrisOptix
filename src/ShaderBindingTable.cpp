#include <graph/ShaderBindingTable.h>
#include <util/math.h>
#include <util/custom_assert.h>
#include <optix_stubs.h>

ShaderBindingTable::ShaderBindingTable(
	const OptixDeviceContext optixDeviceContext,
	const std::vector<OptixProgramGroup>& raygenProgramGroups,
	const std::vector<OptixProgramGroup>& missProgramGroups,
	const std::vector<OptixProgramGroup>& exceptionProgramGroups,
	const std::vector<OptixProgramGroup>& callableProgramGroups,
	const std::vector<OptixProgramGroup>& hitgroupProgramGroups)
{
	std::vector<char> recordsData{};

	static constexpr size_t raygenStride = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
	static constexpr size_t missStride = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
	static constexpr size_t exceptionStride = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
	static constexpr size_t callableStride = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
	static constexpr size_t hitgroupStride = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);

	AddRecords(raygenProgramGroups, raygenStride, recordsData);
	AddRecords(missProgramGroups, missStride, recordsData);
	AddRecords(exceptionProgramGroups, exceptionStride, recordsData);
	AddRecords(callableProgramGroups, callableStride, recordsData);
	AddRecords(hitgroupProgramGroups, hitgroupStride, recordsData);

	m_Buffer.Alloc(recordsData.size());
	m_Buffer.Upload(recordsData.data());

	const size_t raygenOffset = 0;
	const size_t missOffset = raygenOffset + raygenStride * raygenProgramGroups.size();
	const size_t exceptionOffset = missOffset + missStride * missProgramGroups.size();
	const size_t callableOffset = exceptionOffset + exceptionStride * exceptionProgramGroups.size();
	const size_t hitgroupOffset = callableOffset + callableStride * callableProgramGroups.size();

	m_Sbts.resize(raygenProgramGroups.size());
	for (size_t raygenIdx = 0; raygenIdx < m_Sbts.size(); ++raygenIdx)
	{
		OptixShaderBindingTable& sbt = m_Sbts[raygenIdx];

		sbt.raygenRecord = m_Buffer.GetCuPtr(raygenOffset + raygenStride * raygenIdx);

		const auto& [missBase, missStrideBytes, missCount] = GetBaseStrideCount(missProgramGroups, missOffset, missStride);
		sbt.missRecordBase = missBase;
		sbt.missRecordStrideInBytes = missStrideBytes;
		sbt.missRecordCount = missCount;

		// TODO: exception

		const auto& [callableBase, callableStrideBytes, callableCount] = GetBaseStrideCount(callableProgramGroups, callableOffset, callableStride);
		sbt.callablesRecordBase = callableBase;
		sbt.callablesRecordStrideInBytes = callableStrideBytes;
		sbt.callablesRecordCount = callableCount;

		const auto& [hitgroupBase, hitgroupStrideBytes, hitgroupCount] = GetBaseStrideCount(hitgroupProgramGroups, hitgroupOffset, hitgroupStride);
		sbt.hitgroupRecordBase = hitgroupBase;
		sbt.hitgroupRecordStrideInBytes = hitgroupStrideBytes;
		sbt.hitgroupRecordCount = hitgroupCount;
	}
}

ShaderBindingTable::~ShaderBindingTable()
{

}

void ShaderBindingTable::AddRecords(
	const std::vector<OptixProgramGroup>& programGroups,
	const size_t stride,
	std::vector<char>& recordsData)
{
	std::vector<char> tempRecordData(stride);
	char* tempRecordHeaderPtr = tempRecordData.data();
	char* tempRecordDataPtr = tempRecordData.data() + OPTIX_SBT_RECORD_HEADER_SIZE;

	for (const OptixProgramGroup programGroup : programGroups)
	{
		ASSERT_OPTIX(optixSbtRecordPackHeader(programGroup, tempRecordHeaderPtr));
		recordsData.insert(recordsData.end(), tempRecordData.begin(), tempRecordData.end());
	}
}

std::tuple<CUdeviceptr, uint32_t, uint32_t> ShaderBindingTable::GetBaseStrideCount(
	const std::vector<OptixProgramGroup>& programGroups,
	const size_t offset,
	const size_t stride) const
{
	if (programGroups.empty())
	{
		return { 0, 0, 0 };
	}

	return { m_Buffer.GetCuPtr(offset), stride, programGroups.size() };
}
