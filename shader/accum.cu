#include <cuda_runtime.h>
#include <graph/LaunchParams.h>

__constant__ LaunchParams params;

extern "C" __global__ void __raygen__accum()
{

}
