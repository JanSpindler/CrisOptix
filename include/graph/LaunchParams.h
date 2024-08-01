#pragma once

#include <glm/glm.hpp>
#include <graph/Camera.h>
#include <optix.h>
#include <graph/trace.h>
#include <model/Emitter.h>
#include <graph/Reservoir.h>

struct LaunchParams
{
	size_t frameIdx;
	glm::vec3* outputBuffer;
	uint32_t width;
	uint32_t height;
	CameraData cameraData;
	CuBufferView<EmitterData> emitterTable;
	CuBufferView<Reservoir<EmitterSample>> diReservoirs;
	OptixTraversableHandle traversableHandle;
	TraceParameters surfaceTraceParams;
	TraceParameters occlusionTraceParams;
};
