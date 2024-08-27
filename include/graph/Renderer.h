#pragma once

#include <graph/Camera.h>
#include <model/Scene.h>
#include <graph/LaunchParams.h>
#include <graph/CuEvent.h>

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
	// General
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;

	Camera& m_Cam;
	const Scene& m_Scene;
	
	// Launch params
	LaunchParams m_LaunchParams{};
	DeviceBuffer<LaunchParams> m_LaunchParamsBuf = DeviceBuffer<LaunchParams>(1);

	// Buffers
	DeviceBuffer<Reservoir<PrefixPath>> m_PrefixReservoirs{};
	DeviceBuffer<Reservoir<SuffixPath>> m_SuffixReservoirs{};
	DeviceBuffer<RestirGBuffer> m_RestirGBuffers{};
	DeviceBuffer<glm::vec2> m_MotionVectors{};

	// Prefix AS
	PrefixAccelStruct m_PrefixAccelStruct;

	// Pipelines and sbt
	Pipeline m_PrefixGenTempReusePipeline;
	uint32_t m_PrefixGenTempReuseSbtIdx = 0;

	Pipeline m_PrefixSpatialReusePipeline;
	uint32_t m_PrefixSpatialReuseSbtIdx = 0;

	Pipeline m_PrefixStoreEntriesPipeline;
	uint32_t m_PrefixStoreEntriesSbtIdx = 0;

	Pipeline m_SuffixGenTempReusePipeline;
	uint32_t m_SuffixGenTempReuseSbtIdx = 0;

	Pipeline m_SuffixSpatialReusePipeline;
	uint32_t m_SuffixSpatialReuseSbtIdx = 0;

	Pipeline m_FinalGatherPipeline;
	uint32_t m_FinalGatherSbtIdx = 0;

	ShaderBindingTable m_Sbt;

	// Events
	CuEvent m_StartEvent{};
	CuEvent m_PostPrefixGenTempReuseEvent{};
	CuEvent m_PostPrefixSpatialReuseEvent{};
	CuEvent m_PostSuffixGenTempReuseEvent{};
	CuEvent m_PostSuffixSpatialReuseEvent{};
	CuEvent m_PostPrefixStoreEvent{};
	CuEvent m_StopEvent{};

	float m_PrefixGenTempReuseTime = 0.0f;
	float m_PrefixSpatialReuseTime = 0.0f;
	float m_SuffixGenTempReuseTime = 0.0f;
	float m_SuffixSpatialReuseTime = 0.0f;
	float m_PrefixStoreTime	= 0.0f;
	float m_FinalGatherTime = 0.0f;
	float m_TotalTime = 0.0f;

	void RunImGuiSettings();
	void RunImGuiPerformance();
	void RunImGuiPrefixStats();
};
