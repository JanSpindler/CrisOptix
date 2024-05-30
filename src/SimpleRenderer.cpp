#include <graph/SimpleRenderer.h>
#include <optix_stubs.h>

SimpleRenderer::SimpleRenderer(Camera& cam, const Scene& scene) :
	m_Cam(cam),
	m_Scene(scene)
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
	launchParams.surfaceTraceParams.rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
	launchParams.surfaceTraceParams.sbtOffset = 0;
	launchParams.surfaceTraceParams.sbtStride = 1;
	launchParams.surfaceTraceParams.missSbtIdx = 1;
	m_LaunchParamsBuf.Upload(&launchParams);

	ASSERT_OPTIX(optixLaunch(
		m_Scene.GetPipeline().GetHandle(), 
		stream, 
		m_LaunchParamsBuf.GetCuPtr(), 
		m_LaunchParamsBuf.GetByteSize(),
		m_Scene.GetSbt().GetSbt(0),
		width, 
		height, 
		1));
	ASSERT_CUDA(cudaDeviceSynchronize());
}
