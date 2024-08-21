#pragma once

#include <graph/Camera.h>
#include <cuda.h>
#include <model/Scene.h>
#include <graph/DeviceBuffer.h>
#include <graph/LaunchParams.h>
#include <graph/restir/Reservoir.h>
#include <graph/restir/PrefixPath.h>
#include <graph/restir/SuffixPath.h>

class Renderer
{
public:
	Renderer(
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
	DeviceBuffer<RestirGBuffer> m_RestirGBuffers{};
	DeviceBuffer<glm::vec2> m_MotionVectors{};

	Pipeline m_PrefixGenTempReusePipeline;
	uint32_t m_PrefixGenTempReuseSbtIdx = 0;

	Pipeline m_PrefixSpatialReusePipeline;
	uint32_t m_PrefixSpatialReuseSbtIdx = 0;

	Pipeline m_SuffixGenTempReusePipeline;
	uint32_t m_SuffixGenTempReuseSbtIdx = 0;

	Pipeline m_SuffixSpatialReusePipeline;
	uint32_t m_SuffixSpatialReuseSbtIdx = 0;

	ShaderBindingTable m_Sbt;
};
