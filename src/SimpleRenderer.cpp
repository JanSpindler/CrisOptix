#include <graph/SimpleRenderer.h>
#include <optix_stubs.h>

SimpleRenderer::SimpleRenderer(const OptixDeviceContext optixDeviceContext, Camera& cam, const Scene& scene) :
	m_Cam(cam),
	m_Scene(scene),
	m_Pipeline(optixDeviceContext),
	m_Sbt(optixDeviceContext)
{
	const OptixProgramGroup raygenPG = m_Pipeline.AddRaygenShader({ "test.ptx", "__raygen__main" });
	const OptixProgramGroup surfaceMissPG = m_Pipeline.AddMissShader({ "test.ptx", "__miss__main" });
	const OptixProgramGroup occlusionMissPG = m_Pipeline.AddMissShader({ "test.ptx", "__miss__occlusion" });
	
	m_Sbt.AddRaygenEntry(raygenPG);
	m_SurfaceMissIdx = m_Sbt.AddMissEntry(surfaceMissPG);
	m_OcclusionMissIdx = m_Sbt.AddMissEntry(occlusionMissPG);

	m_Scene.AddShader(m_Pipeline, m_Sbt);

	m_Pipeline.CreatePipeline();
	m_Sbt.CreateSBT();
}

void SimpleRenderer::LaunchFrame(
	const CUstream stream,
	glm::vec3* outputBuffer,
	const uint32_t width,
	const uint32_t height)
{
	++m_FrameIdx;
	if (m_Cam.HasChanged()) { m_FrameIdx = 0; }

	LaunchParams launchParams{};
	launchParams.frameIdx = m_FrameIdx;
	launchParams.outputBuffer = outputBuffer;
	launchParams.width = width;
	launchParams.height = height;
	launchParams.traversableHandle = m_Scene.GetTraversableHandle();
	launchParams.cameraData = m_Cam.GetData();
	launchParams.emitterTable = m_Scene.GetEmitterTable();

	launchParams.surfaceTraceParams.rayFlags = OPTIX_RAY_FLAG_NONE;
	launchParams.surfaceTraceParams.sbtOffset = 0;
	launchParams.surfaceTraceParams.sbtStride = 1;
	launchParams.surfaceTraceParams.missSbtIdx = m_SurfaceMissIdx;
	
	launchParams.occlusionTraceParams.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
	launchParams.occlusionTraceParams.sbtOffset = 0;
	launchParams.occlusionTraceParams.sbtStride = 1;
	launchParams.occlusionTraceParams.missSbtIdx = m_OcclusionMissIdx;
	m_LaunchParamsBuf.Upload(&launchParams);

	ASSERT_CUDA(cudaDeviceSynchronize());
	ASSERT_OPTIX(optixLaunch(
		m_Pipeline.GetHandle(), 
		stream, 
		m_LaunchParamsBuf.GetCuPtr(), 
		m_LaunchParamsBuf.GetByteSize(),
		m_Sbt.GetSBT(0),
		width, 
		height, 
		1));
	ASSERT_CUDA(cudaDeviceSynchronize());
}
