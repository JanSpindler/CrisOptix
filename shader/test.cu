#include <optix.h>
#include <cuda_runtime.h>
#include <util/glm_cuda.h>
#include <optix_device.h>

__global__ void __raygen_main()
{
	const glm::uvec3 launchIdx = cuda2glm(optixGetLaunchIndex());
	const glm::uvec3 launchDims = cuda2glm(optixGetLaunchDimensions());
}
