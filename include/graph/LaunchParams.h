#pragma once

#include <glm/glm.hpp>
#include <graph/Camera.h>
#include <optix.h>
#include <graph/trace.h>
#include <graph/Reservoir.h>
#include <model/Emitter.h>
#include <graph/Path.h>

struct LaunchParams
{
	uint32_t random;
	size_t frameIdx;
	glm::vec3* outputBuffer;
	uint32_t width;
	uint32_t height;
	CameraData cameraData;
	bool enableAccum;
	RestirDiParams restirDiParams;
	CuBufferView<EmitterData> emitterTable;
	CuBufferView<Reservoir<EmitterSample>> diReservoirs;
	CuBufferView<Reservoir<Path>> suffixReservoirs;
	OptixTraversableHandle traversableHandle;
	TraceParameters surfaceTraceParams;
	TraceParameters occlusionTraceParams;
};
