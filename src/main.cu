#include <iostream>
#include <Window.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_host.h>
#include <cuda_gl_interop.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <custom_assert.h>
#include <glm/glm.hpp>
#include <OutputBuffer.h>
#include <DeviceBuffer.h>
#include <array>

OptixDeviceContext optixContext = nullptr;

void InitCuda()
{
    std::array<int, 4> cudaGlDevices{};
    uint32_t glDeviceCount = 4;
    cudaGLGetDevices(&glDeviceCount, cudaGlDevices.data(), glDeviceCount, cudaGLDeviceListAll);
    if (glDeviceCount == 0) { Log::Error("No cuda gl capable device found"); }

    //ASSERT_CUDA(cudaSetDevice(cudaGlDevices[0]));
    ASSERT_CUDA(cudaFree(nullptr));
}

void InitOptix()
{
    ASSERT_OPTIX(optixInit());

    const OptixDeviceContextOptions optixContextOptions{};
    optixContext = nullptr;
    ASSERT_OPTIX(optixDeviceContextCreate(0, &optixContextOptions, &optixContext));
}

int main()
{
    std::cout << "Hello there" << std::endl;

    Window::Init(800, 600, false, "CrisOptix");
    InitCuda();
    InitOptix();

    OutputBuffer<glm::u8vec3> outputBuffer(800, 600);

    DeviceBuffer<glm::vec3> hdrBuffer(800 * 600);

    while (!Window::IsClosed())
    {
        // Handle window io
        Window::HandleIO();
        // TODO: handle resize by resizing buffers

        //
        outputBuffer.MapCuda();

        // TODO: trace rays

        // TODO: tone mapping

        //
        outputBuffer.UnmapCuda();

        // Render to window
        Window::Display(outputBuffer.GetPbo());
    }

    Window::Destroy();

    return 0;
}
