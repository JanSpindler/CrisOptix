#pragma once

#include <graph/CuBufferView.h>
#include <graph/restir/Reservoir.h>
#include <graph/restir/PrefixPath.h>
#include <graph/restir/SuffixPath.h>
#include <graph/restir/RestirGBuffer.h>
#include <graph/restir/PrefixAccelStruct.h>
#include <graph/restir/settings.h>

struct RestirParams
{
	// General
	float reconMinRoughness;
	float reconMinDistance;

	// Prefix
	int prefixLen;
	bool prefixEnableTemporal;
	bool prefixEnableSpatial;
	int prefixSpatialCount;

	// Suffix
	bool suffixEnableTemporal;
	bool suffixEnableSpatial;
	int suffixSpatialCount;

	// Final gather
	int gatherN;
	int gatherM;
	float gatherRadius;
	PrefixRadiusType gatherRadiusType;

	// Buffers
	uint8_t frontBufferIdx;
	uint8_t backBufferIdx;
	CuBufferView<Reservoir<PrefixPath>> prefixReservoirs;
	CuBufferView<PrefixPath> canonicalPrefixes;

	CuBufferView<Reservoir<SuffixPath>> suffixReservoirs;
	CuBufferView<SuffixPath> canonicalSuffixes;

	CuBufferView<RestirGBuffer> restirGBuffers;

	// Prefix entries
	OptixTraversableHandle prefixEntriesTraversHandle;
	TraceParameters prefixEntriesTraceParams;
	CuBufferView<OptixAabb> prefixEntryAabbs;
	CuBufferView<PrefixNeighbor> prefixNeighbors;

	// Prefix stats
	bool showPrefixEntries;
	bool showPrefixEntryContrib;
	bool trackPrefixStats;
	CuBufferView<PrefixAccelStruct::Stats> prefixStats;
};
