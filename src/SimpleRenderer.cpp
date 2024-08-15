#include <graph/SimpleRenderer.h>
#include <optix_stubs.h>
#include <imgui.h>
#include <glm/gtc/random.hpp>
#include <limits>
#include <random>

static std::random_device r;
static std::default_random_engine e1(r());
static std::uniform_int_distribution<uint32_t> uniformDist(0, 0xFFFFFFFF);

SimpleRenderer::SimpleRenderer(
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
	m_Pipeline(optixDeviceContext),
	m_Sbt(optixDeviceContext)
{
	//
	m_LaunchParams.neeProb = 0.5f;

	//
	m_LaunchParams.restir.diEnableTemporal = false;
	m_LaunchParams.restir.diCanonicalCount = 1;
	m_LaunchParams.restir.diSpatialCount = 1;
	m_LaunchParams.restir.diSpatialKernelSize = 1;

	m_LaunchParams.restir.minPrefixLen = 2;

	m_LaunchParams.restir.suffixEnableTemporal = false;
	m_LaunchParams.restir.suffixEnableSpatial = false;

	//
	const OptixProgramGroup raygenPG = m_Pipeline.AddRaygenShader({ "test.ptx", "__raygen__main" });
	const OptixProgramGroup surfaceMissPG = m_Pipeline.AddMissShader({ "test.ptx", "__miss__main" });
	const OptixProgramGroup occlusionMissPG = m_Pipeline.AddMissShader({ "test.ptx", "__miss__occlusion" });
	
	//
	const size_t pixelCount = width * height;
	std::vector<Reservoir<PrefixPath>> prefixReservoirs(pixelCount);
	std::vector<Reservoir<SuffixPath>> suffixReservoirs(pixelCount);

	for (size_t idx = 0; idx < pixelCount; ++idx)
	{
		prefixReservoirs[idx] = Reservoir<PrefixPath>();
		suffixReservoirs[idx] = Reservoir<SuffixPath>();
	}

	m_PrefixReservoirs.Alloc(pixelCount);
	m_PrefixReservoirs.Upload(prefixReservoirs.data());

	m_SuffixReservoirs.Alloc(pixelCount);
	m_SuffixReservoirs.Upload(suffixReservoirs.data());

	//
	m_Sbt.AddRaygenEntry(raygenPG);
	m_SurfaceMissIdx = m_Sbt.AddMissEntry(surfaceMissPG);
	m_OcclusionMissIdx = m_Sbt.AddMissEntry(occlusionMissPG);

	m_Scene.AddShader(m_Pipeline, m_Sbt);

	m_Pipeline.CreatePipeline();
	m_Sbt.CreateSBT();
}

void SimpleRenderer::RunImGui()
{
	ImGui::DragFloat("NEE Prob", &m_LaunchParams.neeProb, 0.01f, 0.0f, 1.0f);

	ImGui::Checkbox("Enable Accum", &m_LaunchParams.enableAccum);

	// Restir
	ImGui::Checkbox("Enable Restir", &m_LaunchParams.enableRestir);

	// Restir DI
	ImGui::Text("Restir DI");
	ImGui::InputInt("DI Canonical Count", &m_LaunchParams.restir.diCanonicalCount, 1, 4);
	ImGui::Checkbox("DI Enable Temporal", &m_LaunchParams.restir.diEnableTemporal);
	ImGui::Checkbox("DI Enable Spatial", &m_LaunchParams.restir.diEnableSpatial);
	ImGui::InputInt("DI Spatial Count", &m_LaunchParams.restir.diSpatialCount, 1, 4);
	ImGui::InputInt("DI Spatial Kernel Size", &m_LaunchParams.restir.diSpatialKernelSize, 1, 4);

	// Restir Prefix
	ImGui::Text("Restir Prefix");
	ImGui::InputInt("Prefix Min Len", &m_LaunchParams.restir.minPrefixLen, 1, 1);

	// Restir Suffix
	ImGui::Text("Restir Suffix");
	ImGui::Checkbox("Suffix Enable Temportal", &m_LaunchParams.restir.suffixEnableTemporal);
	ImGui::Checkbox("Suffix Enable Spatial", &m_LaunchParams.restir.suffixEnableSpatial);
}

void SimpleRenderer::LaunchFrame(glm::vec3* outputBuffer)
{
	++m_LaunchParams.frameIdx;
	if (m_Cam.HasChanged() || !m_LaunchParams.enableAccum) { m_LaunchParams.frameIdx = 0; }

	m_LaunchParams.random = uniformDist(e1);
	m_LaunchParams.outputBuffer = outputBuffer;
	m_LaunchParams.width = m_Width;
	m_LaunchParams.height = m_Height;
	m_LaunchParams.traversableHandle = m_Scene.GetTraversableHandle();
	m_LaunchParams.cameraData = m_Cam.GetData();

	m_LaunchParams.emitterTable = m_Scene.GetEmitterTable();

	m_LaunchParams.restir.prefixReservoirs = CuBufferView<Reservoir<PrefixPath>>(m_PrefixReservoirs.GetCuPtr(), m_PrefixReservoirs.GetCount());
	m_LaunchParams.restir.suffixReservoirs = CuBufferView<Reservoir<SuffixPath>>(m_SuffixReservoirs.GetCuPtr(), m_SuffixReservoirs.GetCount());

	m_LaunchParams.surfaceTraceParams.rayFlags = OPTIX_RAY_FLAG_NONE;
	m_LaunchParams.surfaceTraceParams.sbtOffset = 0;
	m_LaunchParams.surfaceTraceParams.sbtStride = 1;
	m_LaunchParams.surfaceTraceParams.missSbtIdx = m_SurfaceMissIdx;
	
	m_LaunchParams.occlusionTraceParams.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
	m_LaunchParams.occlusionTraceParams.sbtOffset = 0;
	m_LaunchParams.occlusionTraceParams.sbtStride = 1;
	m_LaunchParams.occlusionTraceParams.missSbtIdx = m_OcclusionMissIdx;
	m_LaunchParamsBuf.Upload(&m_LaunchParams);

	ASSERT_CUDA(cudaDeviceSynchronize());
	ASSERT_OPTIX(optixLaunch(
		m_Pipeline.GetHandle(), 
		0, 
		m_LaunchParamsBuf.GetCuPtr(), 
		m_LaunchParamsBuf.GetByteSize(),
		m_Sbt.GetSBT(0),
		m_Width, 
		m_Height, 
		1));
	ASSERT_CUDA(cudaDeviceSynchronize());
}

size_t SimpleRenderer::GetFrameIdx() const
{
	return m_LaunchParams.frameIdx;
}
