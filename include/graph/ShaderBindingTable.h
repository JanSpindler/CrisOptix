#pragma once

#include <vector>
#include <memory>
#include <optix.h>
#include <graph/DeviceBuffer.h>

struct HitGroupSBTData
{
    void *shape_data_ptr;
    void *bsdf_data_ptr;
};

template <typename T>
static constexpr std::vector<char> ToVecByte(const T& val)
{
    std::vector<char> vec(sizeof(T));
    std::memcpy(vec.data(), &val, sizeof(T));
    return vec;
}

class ShaderBindingTable
{
public:
    ShaderBindingTable(OptixDeviceContext context);

    // Add an SBT entry without any additional data stored in the SBT, return the index within the respective entry class
    uint32_t AddRaygenEntry  (OptixProgramGroup prog_group) { return AddRaygenEntry(prog_group, std::vector<char>()); }
    uint32_t AddMissEntry    (OptixProgramGroup prog_group) { return AddMissEntry(prog_group, std::vector<char>()); }
    uint32_t AddHitEntry     (OptixProgramGroup prog_group) { return AddHitEntry(prog_group, std::vector<char>()); }
    uint32_t AddCallableEntry(OptixProgramGroup prog_group) { return AddCallableEntry(prog_group, std::vector<char>()); }

    // Add an SBT entry where the contents of sbt_record_data is copied into the data segment of the SBT entry, return the index within the respective entry class
    template <typename T>
    constexpr uint32_t AddRaygenEntry  (OptixProgramGroup prog_group, const T &sbt_record_data) { return AddRaygenEntry(prog_group, std::vector<char>(reinterpret_cast<const char*>(&sbt_record_data), reinterpret_cast<const char*>(&sbt_record_data + 1))); }
    template <typename T>
    constexpr uint32_t AddMissEntry    (OptixProgramGroup prog_group, const T &sbt_record_data) { return AddMissEntry(prog_group, std::vector<char>(reinterpret_cast<const char*>(&sbt_record_data), reinterpret_cast<const char*>(&sbt_record_data + 1))); }
    template <typename T>
    constexpr uint32_t AddHitEntry     (OptixProgramGroup prog_group, const T &sbt_record_data) { return AddHitEntry(prog_group, std::vector<char>(reinterpret_cast<const char*>(&sbt_record_data), reinterpret_cast<const char*>(&sbt_record_data + 1))); }
    template <typename T>
    constexpr uint32_t AddCallableEntry(OptixProgramGroup prog_group, const T &sbt_record_data) { return AddCallableEntry(prog_group, std::vector<char>(reinterpret_cast<const char*>(&sbt_record_data), reinterpret_cast<const char*>(&sbt_record_data + 1))); }

    // Add an SBT entry with custom data stored in an std::vector, return the index within the respective entry class
    uint32_t AddRaygenEntry  (OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data);
    uint32_t AddMissEntry    (OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data);
    uint32_t AddHitEntry     (OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data);
    uint32_t AddCallableEntry(OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data);

    void CreateSBT();

    constexpr const OptixShaderBindingTable *GetSBT(uint32_t raygen_index = 0) const { return raygen_index < m_Sbt.size() ? &m_Sbt[raygen_index] : nullptr; }

private:
    OptixDeviceContext m_Context = nullptr;

    size_t m_MaxRaygenDataSize = 0;
    size_t m_MaxMissDataSize = 0;
    size_t m_MaxHitgroupDataSize = 0;
    size_t m_MaxCallableDataSize = 0;

    std::vector<std::pair<OptixProgramGroup, std::vector<char>>> m_SbtEntriesRaygen{};
    std::vector<std::pair<OptixProgramGroup, std::vector<char>>> m_SbtEntriesMiss{};
    std::vector<std::pair<OptixProgramGroup, std::vector<char>>> m_SbtEntriesHitgroup{};
    std::vector<std::pair<OptixProgramGroup, std::vector<char>>> m_SbtEntriesCallable{};

    DeviceBuffer<char> m_SbtBuffer{};

    std::vector<OptixShaderBindingTable> m_Sbt{};
};
