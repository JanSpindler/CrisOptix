#include <algorithm>
#include <graph/ShaderBindingTable.h>
#include <util/math.h>
#include <optix_stubs.h>

ShaderBindingTable::ShaderBindingTable(OptixDeviceContext context) :
    m_Context { context }
{
}

uint32_t ShaderBindingTable::AddRaygenEntry(OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data)
{
    uint32_t raygen_sbt_index = m_SbtEntriesRaygen.size();
    m_MaxRaygenDataSize = std::max<size_t>(m_MaxRaygenDataSize, sbt_record_custom_data.size());
    m_SbtEntriesRaygen.push_back({prog_group, std::move(sbt_record_custom_data)});
    return raygen_sbt_index;
}

uint32_t ShaderBindingTable::AddMissEntry(OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data)
{
    uint32_t miss_sbt_index = m_SbtEntriesMiss.size();
    m_MaxMissDataSize = std::max<size_t>(m_MaxMissDataSize, sbt_record_custom_data.size());
    m_SbtEntriesMiss.push_back({prog_group, std::move(sbt_record_custom_data)});
    return miss_sbt_index;
}

uint32_t ShaderBindingTable::AddHitEntry(OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data)
{
    uint32_t hitgroup_sbt_index = m_SbtEntriesHitgroup.size();
    m_MaxHitgroupDataSize = std::max<size_t>(m_MaxHitgroupDataSize, sbt_record_custom_data.size());
    m_SbtEntriesHitgroup.push_back({prog_group, std::move(sbt_record_custom_data)});
    return hitgroup_sbt_index;
}

uint32_t ShaderBindingTable::AddCallableEntry(OptixProgramGroup prog_group, std::vector<char> sbt_record_custom_data)
{
    uint32_t callable_sbt_index = m_SbtEntriesCallable.size();
    m_MaxCallableDataSize = std::max<size_t>(m_MaxCallableDataSize, sbt_record_custom_data.size());
    m_SbtEntriesCallable.push_back({prog_group, std::move(sbt_record_custom_data)});
    return callable_sbt_index;
}


void ShaderBindingTable::CreateSBT()
{
    std::vector<char> records_data;

    auto add_records = [&records_data](size_t record_stride, auto &sbt_entries)
    {
        // All entries of the same kind must occupy the same amount of storage
        std::vector<char> temp_record_data(record_stride);
        // Pointer to the header in temporary data
        char *temp_record_header_ptr = temp_record_data.data();
        // Pointer to the data segment in temporary data
        char *temp_record_data_ptr = temp_record_data.data() + OPTIX_SBT_RECORD_HEADER_SIZE;
        for (const auto [ prog_group, data ] : sbt_entries)
        {
            // Write record header
            ASSERT_OPTIX( optixSbtRecordPackHeader(prog_group, temp_record_header_ptr) );
            // Write record data
            std::copy(data.begin(), data.end(), temp_record_data_ptr);

            // Append new record to records_data
            records_data.insert(records_data.end(), temp_record_data.begin(), temp_record_data.end());
        }
    };

    // TODO align the strides!!!

    // Compute the stride of each entry type in the SBT
    auto raygen_stride     = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_MaxRaygenDataSize,   OPTIX_SBT_RECORD_ALIGNMENT);
    auto miss_stride       = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_MaxMissDataSize,     OPTIX_SBT_RECORD_ALIGNMENT);
    auto hitgroup_stride   = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_MaxHitgroupDataSize, OPTIX_SBT_RECORD_ALIGNMENT);
    auto callable_stride   = CeilToMultiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + m_MaxCallableDataSize, OPTIX_SBT_RECORD_ALIGNMENT);

    // Compute the offset for the start of each entry type in the SBT based on the preceeding entries in our records data
    auto raygen_offset   = 0ull;
    auto miss_offset     = raygen_offset   + raygen_stride * m_SbtEntriesRaygen.size();
    auto hitgroup_offset = miss_offset     + miss_stride * m_SbtEntriesMiss.size();
    auto callable_offset = hitgroup_offset + hitgroup_stride * m_SbtEntriesHitgroup.size();
    auto sbt_size        = callable_offset + callable_stride * m_SbtEntriesCallable.size();

    // Reserve enough memory to avoid intermediate allocations
    records_data.reserve(sbt_size);

    // Fill records_data with content!
    add_records(raygen_stride,   m_SbtEntriesRaygen);
    add_records(miss_stride,     m_SbtEntriesMiss);
    add_records(hitgroup_stride, m_SbtEntriesHitgroup);
    add_records(callable_stride, m_SbtEntriesCallable);

    if (records_data.size() != sbt_size)
        throw std::runtime_error("Error encountered while computing SBT data!");

    // Upload records_data to GPU
    m_SbtBuffer.Alloc(records_data.size());
    m_SbtBuffer.Upload(records_data.data());

    // Allocate a separate shader binding table structure for every raygen program.
    m_Sbt.resize(m_SbtEntriesRaygen.size());

    // For all raygen programs...
    for (uint32_t raygen_index = 0; raygen_index < m_Sbt.size(); ++raygen_index)
    {
        // Alias for the sbt we are currently filling...
        auto& sbt = m_Sbt[raygen_index];

        sbt.raygenRecord                = m_SbtBuffer.GetCuPtr(raygen_offset + raygen_index * raygen_stride);

        if (!m_SbtEntriesMiss.empty())
        {
            sbt.missRecordBase              = m_SbtBuffer.GetCuPtr(miss_offset);
            sbt.missRecordStrideInBytes     = miss_stride;
            sbt.missRecordCount             = m_SbtEntriesMiss.size();
        }
        else
        {
            sbt.missRecordBase              = 0;
            sbt.missRecordStrideInBytes     = 0;
            sbt.missRecordCount             = 0;
        }

        if (!m_SbtEntriesHitgroup.empty())
        {
            sbt.hitgroupRecordBase          = m_SbtBuffer.GetCuPtr(hitgroup_offset);
            sbt.hitgroupRecordStrideInBytes = hitgroup_stride;
            sbt.hitgroupRecordCount         = m_SbtEntriesHitgroup.size();
        }
        else
        {
            sbt.hitgroupRecordBase          = 0;
            sbt.hitgroupRecordStrideInBytes = 0;
            sbt.hitgroupRecordCount         = 0;
        }

        if (!m_SbtEntriesCallable.empty())
        {
            sbt.callablesRecordBase          = m_SbtBuffer.GetCuPtr(callable_offset);
            sbt.callablesRecordStrideInBytes = callable_stride;
            sbt.callablesRecordCount         = m_SbtEntriesCallable.size();
        }
        else
        {
            sbt.callablesRecordBase          = 0;
            sbt.callablesRecordStrideInBytes = 0;
            sbt.callablesRecordCount         = 0;
        }
    }
}
