#pragma once

#include <glm/glm.hpp>
#include <graph/Camera.h>
#include <optix.h>
#include <graph/TraceParameters.h>
#include <model/Emitter.h>
#include <graph/restir/RestirParams.h>
#include <graph/restir/settings.h>

struct LaunchParams
{
	uint32_t random;
	size_t frameIdx;

	glm::vec3* outputBuffer;
	uint32_t width;
	uint32_t height;
	CameraData cameraData;

	float neeProb;
	bool enableAccum;
	int maxPathLen;

	RendererType rendererType;
	RestirParams restir;
	
	CuBufferView<glm::vec2> motionVectors;
	CuBufferView<EmitterData> emitterTable;
	CuBufferView<glm::mat4> transforms;
	
	OptixTraversableHandle traversableHandle;
	
	TraceParameters surfaceTraceParams;
	TraceParameters occlusionTraceParams;
};
