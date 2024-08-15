#pragma once

#include <glm/glm.hpp>
#include <graph/Camera.h>
#include <optix.h>
#include <graph/trace.h>
#include <model/Emitter.h>
#include <graph/Path.h>
#include <graph/restir/struct/RestirParams.h>

struct LaunchParams
{
	uint32_t random;
	size_t frameIdx;
	glm::vec3* outputBuffer;
	uint32_t width;
	uint32_t height;
	CameraData cameraData;
	bool enableAccum;
	bool enableRestir;
	RestirParams restir;
	CuBufferView<glm::vec2> motionVectors;
	CuBufferView<EmitterData> emitterTable;
	OptixTraversableHandle traversableHandle;
	TraceParameters surfaceTraceParams;
	TraceParameters occlusionTraceParams;
};
