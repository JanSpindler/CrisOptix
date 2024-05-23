#pragma once

#include <glm/glm.hpp>
#include <graph/Camera.h>
#include <optix.h>

struct LaunchParams
{
	glm::vec3* outputBuffer;
	uint32_t width;
	uint32_t height;
	CameraData cameraData;
	// TODO: emitters
	OptixTraversableHandle traversableHandle;
};
