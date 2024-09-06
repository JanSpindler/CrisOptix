#pragma once

#include <graph/CuBufferView.h>
#include <graph/restir/Reservoir.h>
#include <graph/restir/PrefixPath.h>
#include <graph/restir/SuffixPath.h>
#include <graph/restir/RestirGBuffer.h>
#include <graph/restir/PrefixAccelStruct.h>

struct RestirParams
{
	// Prefix
	int prefixLen;
	
	bool prefixEnableTemporal;
	bool prefixEnableSpatial;

	// Suffix
	bool suffixEnableTemporal;
	bool suffixEnableSpatial;

	// Final gather
	int gatherN;
	int gatherM;
	float gatherRadius;

	// Buffers
	uint8_t frontBufferIdx;
	uint8_t backBufferIdx;
	CuBufferView<Reservoir<PrefixPath>> prefixReservoirs;
	CuBufferView<PrefixPath> canonicalPrefixes;

	CuBufferView<Reservoir<SuffixPath>> suffixReservoirs;
	
	CuBufferView<RestirGBuffer> restirGBuffers;

	// Prefix entries
	OptixTraversableHandle prefixEntriesTraversHandle;
	TraceParameters prefixEntriesTraceParams;
	CuBufferView<OptixAabb> prefixEntryAabbs;
	CuBufferView<PrefixNeighbor> prefixNeighbors;

	// Prefix stats
	bool trackPrefixStats;
	CuBufferView<PrefixAccelStruct::Stats> prefixStats;
};
