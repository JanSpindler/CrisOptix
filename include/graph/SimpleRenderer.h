#pragma once

#include <graph/Camera.h>
#include <cuda.h>
#include <graph/Scene.h>
#include <graph/DeviceBuffer.h>
#include <graph/LaunchParams.h>

class SimpleRenderer
{
public:
	SimpleRenderer(Camera& cam, const Scene& scene, const uint32_t surfaceMissIdx, const uint32_t occlusionMissIdx);

	void LaunchFrame(
		const CUstream stream,
		glm::vec3* outputBuffer, 
		const uint32_t width, 
		const uint32_t height);

private:
	Camera& m_Cam;
	const Scene& m_Scene;
	uint32_t m_SurfaceMissIdx = 0;
	uint32_t m_OcclusionMissIdx = 0;

	DeviceBuffer<LaunchParams> m_LaunchParamsBuf = DeviceBuffer<LaunchParams>(1);
};
