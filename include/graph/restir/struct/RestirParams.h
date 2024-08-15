#pragma once

#include <graph/CuBufferView.h>
#include <graph/restir/struct/Reservoir.h>
#include <graph/restir/struct/PrefixPath.h>
#include <graph/restir/struct/SuffixPath.h>

struct RestirParams
{
	// Restir DI
	int diCanonicalCount;
	bool diEnableTemporal;
	bool diEnableSpatial;
	int diSpatialCount;
	int diSpatialKernelSize;

	// Prefix
	bool adaptivePrefixLength;
	int minPrefixLen;
	uint32_t maxReconLength;

	// Suffix
	bool suffixEnableTemporal;
	bool suffixEnableSpatial;

	//CuBufferView<PathReservoir> pathReservoirs;
	CuBufferView<Reservoir<PrefixPath>> prefixReservoirs;
	CuBufferView<Reservoir<SuffixPath>> suffixReservoirs;
	//CuBufferView<SurfaceInteraction> primaryInteractions;
	//CuBufferView<PrefixGBuffer> prefixGBuffers;
};
