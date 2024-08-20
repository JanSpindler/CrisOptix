#pragma once

#include <graph/CuBufferView.h>
#include <graph/restir/Reservoir.h>
#include <graph/restir/PrefixPath.h>
#include <graph/restir/SuffixPath.h>

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

	bool prefixEnableTemporal;
	bool prefixEnableSpatial;

	// Suffix
	bool suffixEnableTemporal;
	bool suffixEnableSpatial;

	//CuBufferView<PathReservoir> pathReservoirs;
	CuBufferView<Reservoir<PrefixPath>> prefixReservoirs;
	CuBufferView<Reservoir<SuffixPath>> suffixReservoirs;
	//CuBufferView<SurfaceInteraction> primaryInteractions;
	//CuBufferView<PrefixGBuffer> prefixGBuffers;
};
