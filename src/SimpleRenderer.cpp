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
	m_DiReservoirs(width * height),
	m_Pipeline(optixDeviceContext),
	m_Sbt(optixDeviceContext)
{
	m_LaunchParams.restirDiParams.enableTemporal = true;
	m_LaunchParams.restirDiParams.canonicalCount = 1;
	m_LaunchParams.restirDiParams.spatialCount = 1;
	m_LaunchParams.restirDiParams.spatialKernelSize = 1;

	const OptixProgramGroup raygenPG = m_Pipeline.AddRaygenShader({ "test.ptx", "__raygen__main" });
	const OptixProgramGroup surfaceMissPG = m_Pipeline.AddMissShader({ "test.ptx", "__miss__main" });
	const OptixProgramGroup occlusionMissPG = m_Pipeline.AddMissShader({ "test.ptx", "__miss__occlusion" });
	
	const size_t pixelCount = width * height;
	std::vector<Reservoir<EmitterSample>> reservoirs(width * height);
	for (size_t idx = 0; idx < pixelCount; ++idx)
	{
		reservoirs[idx] = { {}, 0.0f, 0 };
	}

	m_DiReservoirs.Alloc(pixelCount);
	m_DiReservoirs.Upload(reservoirs.data());

	m_Sbt.AddRaygenEntry(raygenPG);
	m_SurfaceMissIdx = m_Sbt.AddMissEntry(surfaceMissPG);
	m_OcclusionMissIdx = m_Sbt.AddMissEntry(occlusionMissPG);

	m_Scene.AddShader(m_Pipeline, m_Sbt);

	m_Pipeline.CreatePipeline();
	m_Sbt.CreateSBT();
}

void SimpleRenderer::RunImGui()
{
	ImGui::Checkbox("Enable Accum", &m_LaunchParams.enableAccum);
	ImGui::Checkbox("Enable Temporal", &m_LaunchParams.restirDiParams.enableTemporal);
	ImGui::InputInt("Canonical Count", &m_LaunchParams.restirDiParams.canonicalCount, 1, 4);
	ImGui::InputInt("Spatial Count", &m_LaunchParams.restirDiParams.spatialCount);
	ImGui::InputInt("Spatial Kernel Size", &m_LaunchParams.restirDiParams.spatialKernelSize);
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
	m_LaunchParams.diReservoirs = CuBufferView<Reservoir<EmitterSample>>(m_DiReservoirs.GetCuPtr(), m_DiReservoirs.GetCount());

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
