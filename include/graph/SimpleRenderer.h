#pragma once

#include <graph/Camera.h>
#include <cuda.h>
#include <graph/Scene.h>
#include <graph/DeviceBuffer.h>
#include <graph/LaunchParams.h>

class SimpleRenderer
{
public:
	SimpleRenderer(Camera& cam, const Scene& scene);

	void LaunchFrame(
		const CUstream stream,
		glm::vec3* outputBuffer, 
		const uint32_t width, 
		const uint32_t height);

private:
	Camera& m_Cam;
	const Scene& m_Scene;
	DeviceBuffer<LaunchParams> m_LaunchParamsBuf = DeviceBuffer<LaunchParams>(1);
};
