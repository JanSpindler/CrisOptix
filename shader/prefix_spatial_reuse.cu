#include <cuda_runtime.h>
#include <optix_device.h>
#include <util/glm_cuda.h>
#include <graph/LaunchParams.h>
#include <graph/restir/path_gen.h>

__constant__ LaunchParams params;

extern "C" __global__ void __raygen__prefix_spatial_reuse()
{
}
