#include <graph/Renderer.h>
#include <optix_stubs.h>
#include <imgui.h>
#include <glm/gtc/random.hpp>
#include <limits>
#include <random>
#include <algorithm>

static std::random_device r{};
static std::default_random_engine e1(r());
static std::uniform_int_distribution<uint32_t> uniformDist(0, 0xFFFFFFFF);

Renderer::Renderer(
	const uint32_t width,
	const uint32_t height,
	const OptixDeviceContext optixDeviceContext,
	Camera& cam,
	const Scene& scene)
	:
	m_Width(width),
	m_Height(height),
	m_Cam(cam),
	m_Scene(scene),
	m_PrefixAccelStruct(width * height, 0, optixDeviceContext),
	m_PrefixGenTempReusePipeline(optixDeviceContext),
	m_PrefixSpatialReusePipeline(optixDeviceContext),
	m_PrefixStoreEntriesPipeline(optixDeviceContext),
	m_SuffixGenTempReusePipeline(optixDeviceContext),
	m_SuffixSpatialReusePipeline(optixDeviceContext),
	m_FinalGatherPipeline(optixDeviceContext),
	m_Sbt(optixDeviceContext)
{
	//
	m_LaunchParams.neeProb = 0.1f;
	
	//
	m_LaunchParams.restir.prefixLen = 2;
	m_LaunchParams.restir.prefixEnableTemporal = false;
	m_LaunchParams.restir.prefixEnableSpatial = false;
	m_LaunchParams.restir.prefixSpatialCount = 1;

	m_LaunchParams.restir.suffixEnableTemporal = false;
	m_LaunchParams.restir.suffixEnableSpatial = false;
	m_LaunchParams.restir.suffixSpatialCount = 1;

	m_LaunchParams.restir.gatherN = 1;
	m_LaunchParams.restir.gatherM = 1;
	m_LaunchParams.restir.gatherRadius = 0.01f;

	m_LaunchParams.restir.trackPrefixStats = false;

	//
	m_LaunchParams.surfaceTraceParams.rayFlags = OPTIX_RAY_FLAG_NONE;
	m_LaunchParams.surfaceTraceParams.sbtOffset = 0;
	m_LaunchParams.surfaceTraceParams.sbtStride = 1;
	//m_LaunchParams.surfaceTraceParams.missSbtIdx;

	m_LaunchParams.occlusionTraceParams.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
	m_LaunchParams.occlusionTraceParams.sbtOffset = 0;
	m_LaunchParams.occlusionTraceParams.sbtStride = 1;
	//m_LaunchParams.occlusionTraceParams.missSbtIdx;
	
	m_LaunchParams.restir.prefixEntriesTraceParams.rayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT;
	m_LaunchParams.restir.prefixEntriesTraceParams.sbtOffset = 0;
	m_LaunchParams.restir.prefixEntriesTraceParams.sbtStride = 1;
	//m_LaunchParams.restir.prefixEntriesTraceParams.missSbtIdx;

	// Pipelines
	// Miss
	const ShaderEntryPointDesc surfaceMissEntryPointDesc = { "miss.ptx", "__miss__main" };
	const ShaderEntryPointDesc occlusionMissEntryPointDesc = { "miss.ptx", "__miss__occlusion" };

	const OptixProgramGroupDesc surfaceMissPgDesc = Pipeline::GetPgDesc(surfaceMissEntryPointDesc, OPTIX_PROGRAM_GROUP_KIND_MISS, optixDeviceContext);
	const OptixProgramGroupDesc occlusionMissPgDesc = Pipeline::GetPgDesc(occlusionMissEntryPointDesc, OPTIX_PROGRAM_GROUP_KIND_MISS, optixDeviceContext);

	const OptixProgramGroup surfaceMissPG = m_PrefixGenTempReusePipeline.AddMissShader(surfaceMissEntryPointDesc);
	const OptixProgramGroup occlusionMissPG = m_PrefixGenTempReusePipeline.AddMissShader(occlusionMissEntryPointDesc);

	m_LaunchParams.surfaceTraceParams.missSbtIdx = m_Sbt.AddMissEntry(surfaceMissPG);
	m_LaunchParams.occlusionTraceParams.missSbtIdx = m_Sbt.AddMissEntry(occlusionMissPG);

	// Prefix gen and temp reuse
	const OptixProgramGroup prefixGenTempReusePG = m_PrefixGenTempReusePipeline.AddRaygenShader({ "prefix_gen_temp_reuse.ptx", "__raygen__prefix_gen_temp_reuse" });
	m_PrefixGenTempReuseSbtIdx = m_Sbt.AddRaygenEntry(prefixGenTempReusePG);

	m_Scene.AddShader(m_PrefixGenTempReusePipeline, m_Sbt);
	m_PrefixGenTempReusePipeline.CreatePipeline();

	// Prefix spatial reuse
	const OptixProgramGroup prefixSpatialReusePG = m_PrefixSpatialReusePipeline.AddRaygenShader({ "prefix_spatial_reuse.ptx", "__raygen__prefix_spatial_reuse" });
	m_PrefixSpatialReuseSbtIdx = m_Sbt.AddRaygenEntry(prefixSpatialReusePG);

	m_PrefixSpatialReusePipeline.AddProgramGroup(surfaceMissPgDesc, surfaceMissPG);
	m_PrefixSpatialReusePipeline.AddProgramGroup(occlusionMissPgDesc, occlusionMissPG);

	m_Scene.AddShader(m_PrefixSpatialReusePipeline, m_Sbt);
	m_PrefixSpatialReusePipeline.CreatePipeline();

	// Prefix store entries
	// TODO: Implement as normal cuda kernel? Uses no optix features
	const OptixProgramGroup prefixStoreEntriesPG = m_PrefixStoreEntriesPipeline.AddRaygenShader({ "prefix_store_entries.ptx", "__raygen__prefix_store_entries" });
	m_PrefixStoreEntriesSbtIdx = m_Sbt.AddRaygenEntry(prefixStoreEntriesPG);

	m_PrefixStoreEntriesPipeline.CreatePipeline();

	// Suffix gen and temp reuse
	const OptixProgramGroup suffixGenTempReusePG = m_SuffixGenTempReusePipeline.AddRaygenShader({ "suffix_gen_temp_reuse.ptx", "__raygen__suffix_gen_temp_reuse" });
	m_SuffixGenTempReuseSbtIdx = m_Sbt.AddRaygenEntry(suffixGenTempReusePG);

	m_SuffixGenTempReusePipeline.AddProgramGroup(surfaceMissPgDesc, surfaceMissPG);
	m_SuffixGenTempReusePipeline.AddProgramGroup(occlusionMissPgDesc, occlusionMissPG);

	m_Scene.AddShader(m_SuffixGenTempReusePipeline, m_Sbt);
	m_SuffixGenTempReusePipeline.CreatePipeline();

	// Suffix spatial reuse
	const OptixProgramGroup suffixSpatialReusePG = m_SuffixSpatialReusePipeline.AddRaygenShader({ "suffix_spatial_reuse.ptx", "__raygen__suffix_spatial_reuse" });
	m_SuffixSpatialReuseSbtIdx = m_Sbt.AddRaygenEntry(suffixSpatialReusePG);

	m_SuffixSpatialReusePipeline.AddProgramGroup(surfaceMissPgDesc, surfaceMissPG);
	m_SuffixSpatialReusePipeline.AddProgramGroup(occlusionMissPgDesc, occlusionMissPG);

	m_Scene.AddShader(m_SuffixSpatialReusePipeline, m_Sbt);
	m_SuffixSpatialReusePipeline.CreatePipeline();

	// Final gather
	const OptixProgramGroup finalGatherPG = m_FinalGatherPipeline.AddRaygenShader({ "final_gather.ptx", "__raygen__final_gather" });
	m_FinalGatherSbtIdx = m_Sbt.AddRaygenEntry(finalGatherPG);

	const OptixProgramGroup prefixEntryPG = m_FinalGatherPipeline.AddProceduralHitGroupShader({ "final_gather.ptx", "__intersection__prefix_entry" }, {}, {});
	m_PrefixAccelStruct.SetSbtOffset(m_Sbt.AddHitEntry(prefixEntryPG));
	
	const OptixProgramGroup prefixEntryMissPG = m_FinalGatherPipeline.AddMissShader({ "miss.ptx", "__miss__prefix_entry" });
	m_LaunchParams.restir.prefixEntriesTraceParams.missSbtIdx = m_Sbt.AddMissEntry(prefixEntryMissPG);

	m_FinalGatherPipeline.AddProgramGroup(surfaceMissPgDesc, surfaceMissPG);
	m_FinalGatherPipeline.AddProgramGroup(occlusionMissPgDesc, occlusionMissPG);

	m_Scene.AddShader(m_FinalGatherPipeline, m_Sbt);
	m_FinalGatherPipeline.CreatePipeline();

	// TODO: Make m_Scene.AddShader() more efficient (duplicates sbt entries for every pipeline)

	// Sbt
	m_Sbt.CreateSBT();

	// LaunchParams buffers
	m_LaunchParams.restir.frontBufferIdx = 0;
	m_LaunchParams.restir.backBufferIdx = 0;

	const size_t pixelCount = width * height;
	std::vector<Reservoir<PrefixPath>> prefixReservoirs(pixelCount * 2);
	std::vector<PrefixPath> canonicalPrefixes(pixelCount);

	std::vector<Reservoir<SuffixPath>> suffixReservoirs(pixelCount * 2);

	std::vector<RestirGBuffer> restirGBuffers(pixelCount);
	std::vector<glm::vec2> motionVectors(pixelCount);

	for (size_t idx = 0; idx < pixelCount; ++idx)
	{
		prefixReservoirs[idx * 2 + 0] = Reservoir<PrefixPath>();
		prefixReservoirs[idx * 2 + 1] = Reservoir<PrefixPath>();
		canonicalPrefixes[idx] = PrefixPath();

		suffixReservoirs[idx * 2 + 0] = Reservoir<SuffixPath>();
		suffixReservoirs[idx * 2 + 1] = Reservoir<SuffixPath>();
		
		restirGBuffers[idx] = RestirGBuffer();
		motionVectors[idx] = glm::vec2(0.0f);
	}

	m_PrefixReservoirs.Alloc(pixelCount * 2);
	m_PrefixReservoirs.Upload(prefixReservoirs.data());

	m_CanonicalPrefixes.Alloc(pixelCount);
	m_CanonicalPrefixes.Upload(canonicalPrefixes.data());

	m_SuffixReservoirs.Alloc(pixelCount * 2);
	m_SuffixReservoirs.Upload(suffixReservoirs.data());

	m_RestirGBuffers.Alloc(pixelCount);
	m_RestirGBuffers.Upload(restirGBuffers.data());

	m_MotionVectors.Alloc(pixelCount);
	m_MotionVectors.Upload(motionVectors.data());
}

void Renderer::RunImGui()
{
	RunImGuiSettings();
	RunImGuiPerformance();
	RunImGuiPrefixStats();
}

void Renderer::LaunchFrame(glm::vec3* outputBuffer)
{
	// Launch params
	++m_LaunchParams.frameIdx;
	if (m_Cam.HasChanged() || !m_LaunchParams.enableAccum) { m_LaunchParams.frameIdx = 0; }

	m_LaunchParams.random = uniformDist(e1);
	m_LaunchParams.outputBuffer = outputBuffer;
	m_LaunchParams.width = m_Width;
	m_LaunchParams.height = m_Height;
	m_LaunchParams.traversableHandle = m_Scene.GetTraversableHandle();
	m_LaunchParams.cameraData = m_Cam.GetData();

	if (m_LaunchParams.restir.frontBufferIdx == 0)
	{
		m_LaunchParams.restir.frontBufferIdx = 1;
		m_LaunchParams.restir.backBufferIdx = 0;
	}
	else
	{
		m_LaunchParams.restir.frontBufferIdx = 0;
		m_LaunchParams.restir.backBufferIdx = 1;
	}

	m_LaunchParams.restir.prefixReservoirs = CuBufferView<Reservoir<PrefixPath>>(m_PrefixReservoirs.GetCuPtr(), m_PrefixReservoirs.GetCount());
	m_LaunchParams.restir.canonicalPrefixes = CuBufferView<PrefixPath>(m_CanonicalPrefixes.GetCuPtr(), m_CanonicalPrefixes.GetCount());

	m_LaunchParams.restir.suffixReservoirs = CuBufferView<Reservoir<SuffixPath>>(m_SuffixReservoirs.GetCuPtr(), m_SuffixReservoirs.GetCount());
	
	m_LaunchParams.restir.restirGBuffers = CuBufferView<RestirGBuffer>(m_RestirGBuffers.GetCuPtr(), m_RestirGBuffers.GetCount());

	m_LaunchParams.restir.prefixEntryAabbs = m_PrefixAccelStruct.GetAabbBufferView();
	m_LaunchParams.restir.prefixNeighbors = m_PrefixAccelStruct.GetPrefixNeighborBufferView();
	m_LaunchParams.restir.prefixStats = m_PrefixAccelStruct.GetStatsBufferView();

	m_LaunchParams.motionVectors = CuBufferView<glm::vec2>(m_MotionVectors.GetCuPtr(), m_MotionVectors.GetCount());

	m_LaunchParams.emitterTable = m_Scene.GetEmitterTable();

	m_LaunchParamsBuf.Upload(&m_LaunchParams);

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());

	// Record start event
	m_StartEvent.Record();

	// Prefix gen and temporal reuse
	ASSERT_OPTIX(optixLaunch(
		m_PrefixGenTempReusePipeline.GetHandle(), 
		0, 
		m_LaunchParamsBuf.GetCuPtr(), 
		m_LaunchParamsBuf.GetByteSize(),
		m_Sbt.GetSBT(m_PrefixGenTempReuseSbtIdx),
		m_Width, 
		m_Height, 
		1));
	m_PostPrefixGenTempReuseEvent.Record();

	// Restir
	if (m_LaunchParams.enableRestir)
	{
		// Prefix spatial reuse
		if (m_LaunchParams.restir.prefixEnableSpatial && m_LaunchParams.restir.prefixSpatialCount > 0)
		{
			ASSERT_OPTIX(optixLaunch(
				m_PrefixSpatialReusePipeline.GetHandle(),
				0,
				m_LaunchParamsBuf.GetCuPtr(),
				m_LaunchParamsBuf.GetByteSize(),
				m_Sbt.GetSBT(m_PrefixSpatialReuseSbtIdx),
				m_Width,
				m_Height,
				1));
		}
		m_PostPrefixSpatialReuseEvent.Record();

		// Suffix gen and temp reuse
		ASSERT_OPTIX(optixLaunch(
			m_SuffixGenTempReusePipeline.GetHandle(),
			0,
			m_LaunchParamsBuf.GetCuPtr(),
			m_LaunchParamsBuf.GetByteSize(),
			m_Sbt.GetSBT(m_SuffixGenTempReuseSbtIdx),
			m_Width,
			m_Height,
			1));
		m_PostSuffixGenTempReuseEvent.Record();

		// Suffix spatial reuse
		if (m_LaunchParams.restir.suffixEnableSpatial && m_LaunchParams.restir.suffixSpatialCount > 0)
		{
			ASSERT_OPTIX(optixLaunch(
				m_SuffixSpatialReusePipeline.GetHandle(),
				0,
				m_LaunchParamsBuf.GetCuPtr(),
				m_LaunchParamsBuf.GetByteSize(),
				m_Sbt.GetSBT(m_SuffixSpatialReuseSbtIdx),
				m_Width,
				m_Height,
				1));
		}
		m_PostSuffixSpatialReuseEvent.Record();

		// Prefix store entries
		{
			// Store entries in buffer
			ASSERT_OPTIX(optixLaunch(
				m_PrefixStoreEntriesPipeline.GetHandle(),
				0,
				m_LaunchParamsBuf.GetCuPtr(),
				m_LaunchParamsBuf.GetByteSize(),
				m_Sbt.GetSBT(m_PrefixStoreEntriesSbtIdx),
				m_Width,
				m_Height,
				1));

			// Sync
			ASSERT_CUDA(cudaDeviceSynchronize());

			// Rebuild acceleration structure
			m_PrefixAccelStruct.Rebuild(m_LaunchParams.restir.gatherM - 1);

			// Copy traversable handle into launch params and re-upload
			// TODO: Optimize by constant CUdeviceptr to buffer containing handle?
			m_LaunchParams.restir.prefixEntriesTraversHandle = m_PrefixAccelStruct.GetTlas();
			m_LaunchParams.restir.prefixNeighbors = m_PrefixAccelStruct.GetPrefixNeighborBufferView();
			m_LaunchParamsBuf.Upload(&m_LaunchParams);

			// Sync after buffer upload
			ASSERT_CUDA(cudaDeviceSynchronize());
		}
		m_PostPrefixStoreEvent.Record();

		// Final gather
		ASSERT_OPTIX(optixLaunch(
			m_FinalGatherPipeline.GetHandle(),
			0,
			m_LaunchParamsBuf.GetCuPtr(),
			m_LaunchParamsBuf.GetByteSize(),
			m_Sbt.GetSBT(m_FinalGatherSbtIdx),
			m_Width,
			m_Height,
			1));
	}
	// If not restir
	else
	{
		// Record all unused restir events so they will produce 0ms delta times
		m_PostPrefixSpatialReuseEvent.Record();
		m_PostSuffixGenTempReuseEvent.Record();
		m_PostSuffixSpatialReuseEvent.Record();
		m_PostPrefixStoreEvent.Record();
	}
	
	// Ensure that stop event is recorded at the end
	m_StopEvent.Record();
	m_StopEvent.Sync();

	// Store time
	m_PrefixGenTempReuseTime = CuEvent::GetElapsedTimeMs(m_StartEvent, m_PostPrefixGenTempReuseEvent);
	m_PrefixSpatialReuseTime = CuEvent::GetElapsedTimeMs(m_PostPrefixGenTempReuseEvent, m_PostPrefixSpatialReuseEvent);
	m_SuffixGenTempReuseTime = CuEvent::GetElapsedTimeMs(m_PostPrefixSpatialReuseEvent, m_PostSuffixGenTempReuseEvent);
	m_SuffixSpatialReuseTime = CuEvent::GetElapsedTimeMs(m_PostSuffixGenTempReuseEvent, m_PostSuffixSpatialReuseEvent);
	m_PrefixStoreTime = CuEvent::GetElapsedTimeMs(m_PostSuffixSpatialReuseEvent, m_PostPrefixStoreEvent);
	m_FinalGatherTime = CuEvent::GetElapsedTimeMs(m_PostPrefixStoreEvent, m_StopEvent);
	m_TotalTime = CuEvent::GetElapsedTimeMs(m_StartEvent, m_StopEvent);

	// Sync
	ASSERT_CUDA(cudaDeviceSynchronize());
}

size_t Renderer::GetFrameIdx() const
{
	return m_LaunchParams.frameIdx;
}

void Renderer::RunImGuiSettings()
{
	//
	ImGui::Begin("Renderer Settings");

	// General rendering
	ImGui::DragFloat("NEE Prob", &m_LaunchParams.neeProb, 0.01f, 0.0f, 1.0f);
	ImGui::InputInt("NEE Tries", &m_LaunchParams.neeTries, 1, 4);
	m_LaunchParams.neeTries = std::max<int>(1, m_LaunchParams.neeTries);
	ImGui::Checkbox("Enable Accum", &m_LaunchParams.enableAccum);

	// Restir
	ImGui::Checkbox("Enable Restir", &m_LaunchParams.enableRestir);

	// Restir Prefix
	ImGui::Text("Restir Prefix");
	ImGui::InputInt("Prefix Min Len", &m_LaunchParams.restir.prefixLen, 1, 1);
	m_LaunchParams.restir.prefixLen = std::max<int>(1, m_LaunchParams.restir.prefixLen);
	ImGui::Checkbox("Prefix Enable Temporal", &m_LaunchParams.restir.prefixEnableTemporal);
	ImGui::Checkbox("Prefix Enable Spatial", &m_LaunchParams.restir.prefixEnableSpatial);
	ImGui::InputInt("Prefix Spatial Count", &m_LaunchParams.restir.prefixSpatialCount);
	m_LaunchParams.restir.prefixSpatialCount = std::max<int>(0, m_LaunchParams.restir.prefixSpatialCount);

	// Restir Suffix
	ImGui::Text("Restir Suffix");
	ImGui::Checkbox("Suffix Enable Temporal", &m_LaunchParams.restir.suffixEnableTemporal);
	ImGui::Checkbox("Suffix Enable Spatial", &m_LaunchParams.restir.suffixEnableSpatial);
	ImGui::InputInt("Suffix Spatial Count", &m_LaunchParams.restir.suffixSpatialCount);
	m_LaunchParams.restir.suffixSpatialCount = std::max<int>(0, m_LaunchParams.restir.suffixSpatialCount);

	// Restir final gather
	ImGui::Text("Restir Final Gather");
	ImGui::InputInt("Final Gather N", &m_LaunchParams.restir.gatherN, 1, 4);
	m_LaunchParams.restir.gatherN = std::max<int>(1, m_LaunchParams.restir.gatherN);
	ImGui::InputInt("Final Gather M", &m_LaunchParams.restir.gatherM, 1, 4);
	m_LaunchParams.restir.gatherM = std::max<int>(1, m_LaunchParams.restir.gatherM);
	ImGui::DragFloat("Final Gather Radius", &m_LaunchParams.restir.gatherRadius, 0.001f, 0.0f, 1.0f);

	//
	ImGui::End();
}

void Renderer::RunImGuiPerformance()
{
	ImGui::Begin("Renderer Performance");

	ImGui::Text("Prefix Gen Temp Reuse: %fms", m_PrefixGenTempReuseTime);
	ImGui::Text("Prefix Spatial Reuse: %fms", m_PrefixSpatialReuseTime);
	ImGui::Text("Suffix Gen Temp Reuse: %fms", m_SuffixGenTempReuseTime);
	ImGui::Text("Suffix Spatial Reuse:  %fms", m_SuffixSpatialReuseTime);
	ImGui::Text("Prefix Store: %fms", m_PrefixStoreTime);
	ImGui::Text("Final Gather: %fms", m_FinalGatherTime);
	ImGui::Text("Total Time: %fms", m_TotalTime);

	ImGui::End();
}

void Renderer::RunImGuiPrefixStats()
{
	ImGui::Begin("Renderer Prefix Stats");

	const PrefixAccelStruct::Stats stats = m_PrefixAccelStruct.GetStats();
	const float avgNeighCount = static_cast<float>(stats.totalNeighCount) / static_cast<float>(m_Width * m_Height * (m_LaunchParams.restir.gatherN - 1));

	ImGui::Checkbox("Track Prefix Stats", &m_LaunchParams.restir.trackPrefixStats);
	ImGui::Text("Min neighbor count: %d", stats.minNeighCount);
	ImGui::Text("Max neighbor count: %d", stats.maxNeighCount);
	ImGui::Text("Avg neighbor count: %f", avgNeighCount);

	ImGui::End();

	m_PrefixAccelStruct.ResetStats();
}
