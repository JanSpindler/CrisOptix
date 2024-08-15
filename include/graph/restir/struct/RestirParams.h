#pragma once

#include <graph/CuBufferView.h>
#include <graph/restir/struct/Reservoir.h>
#include <graph/restir/struct/PrefixPath.h>
#include <graph/restir/struct/SuffixPath.h>

struct RestirParams
{
	int canonicalCount;

	bool enableTemporal;

	bool enableSpatial;
	int spatialCount;
	int spatialKernelSize;

	bool adaptivePrefixLength;
	uint32_t minPrefixLen;
	uint32_t maxReconLength;

	//CuBufferView<PathReservoir> pathReservoirs;
	CuBufferView<Reservoir<PrefixPath>> prefixReservoirs;
	CuBufferView<Reservoir<SuffixPath>> suffixReservoirs;
	//CuBufferView<SurfaceInteraction> primaryInteractions;
	//CuBufferView<PrefixGBuffer> prefixGBuffers;
};
