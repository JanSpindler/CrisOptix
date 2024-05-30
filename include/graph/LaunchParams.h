#pragma once

#include <glm/glm.hpp>
#include <graph/Camera.h>
#include <optix.h>
#include <graph/trace.h>

struct LaunchParams
{
	glm::vec3* outputBuffer;
	uint32_t width;
	uint32_t height;
	CameraData cameraData;
	OptixTraversableHandle traversableHandle;
	TraceParameters surfaceTraceParams;
};
