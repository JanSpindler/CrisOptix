#pragma once

#include <graph/Camera.h>
#include <cuda.h>
#include <model/Scene.h>
#include <graph/DeviceBuffer.h>
#include <graph/LaunchParams.h>
#include <graph/restir/struct/Reservoir.h>
#include <graph/restir/struct/PrefixPath.h>
#include <graph/restir/struct/SuffixPath.h>

class SimpleRenderer
{
public:
	SimpleRenderer(
		const uint32_t width,
		const uint32_t height,
		const OptixDeviceContext optixDeviceContext, 
		Camera& cam, 
		const Scene& scene);

	void RunImGui();
	void LaunchFrame(glm::vec3* outputBuffer);

	size_t GetFrameIdx() const;

private:
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;

	Camera& m_Cam;
	const Scene& m_Scene;
	uint32_t m_SurfaceMissIdx = 0;
	uint32_t m_OcclusionMissIdx = 0;

	LaunchParams m_LaunchParams{};
	DeviceBuffer<LaunchParams> m_LaunchParamsBuf = DeviceBuffer<LaunchParams>(1);

	DeviceBuffer<Reservoir<PrefixPath>> m_PrefixReservoirs{};
	DeviceBuffer<Reservoir<SuffixPath>> m_SuffixReservoirs{};

	Pipeline m_Pipeline;
	ShaderBindingTable m_Sbt;
};
