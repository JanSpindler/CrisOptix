#include <graph/SimpleRenderer.h>
#include <optix_stubs.h>

SimpleRenderer::SimpleRenderer(Camera& cam, const Scene& scene, const uint32_t surfaceMissIdx, const uint32_t occlusionMissIdx) :
	m_Cam(cam),
	m_Scene(scene),
	m_SurfaceMissIdx(surfaceMissIdx),
	m_OcclusionMissIdx(occlusionMissIdx)
{
}

void SimpleRenderer::LaunchFrame(
	const CUstream stream,
	glm::vec3* outputBuffer,
	const uint32_t width,
	const uint32_t height)
{
	LaunchParams launchParams{};
	launchParams.outputBuffer = outputBuffer;
	launchParams.width = width;
	launchParams.height = height;
	launchParams.traversableHandle = m_Scene.GetTraversableHandle();
	launchParams.cameraData = m_Cam.GetData();
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
		m_Scene.GetPipeline().GetHandle(), 
		stream, 
		m_LaunchParamsBuf.GetCuPtr(), 
		m_LaunchParamsBuf.GetByteSize(),
		m_Scene.GetSbt().GetSBT(0),
		width, 
		height, 
		1));
	ASSERT_CUDA(cudaDeviceSynchronize());
}
