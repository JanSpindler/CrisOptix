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

OptixDeviceContext optixContext = nullptr;

void InitOptix()
{
    if (cudaFree(0) != CUDA_SUCCESS) { exit(1); }
    if (optixInit() != OPTIX_SUCCESS) { exit(1); }

    const OptixDeviceContextOptions optixContextOptions{};
    optixContext = nullptr;
    ASSERT_OPTIX(optixDeviceContextCreate(0, &optixContextOptions, &optixContext));
}

int main()
{
    std::cout << "Hello there" << std::endl;

    Window::Init(800, 600, false, "CrisOptix");
    InitOptix();

    OutputBuffer<glm::u8vec3> outputBuffer(800, 600);

    DeviceBuffer<glm::vec3> hdrBuffer(800 * 600);

    while (!Window::IsClosed())
    {
        //
        outputBuffer.MapCuda();

        // Trace rays
        // TODO: trace rays

        // Tone mapping

        // Render to window
        Window::Update();
    }

    Window::Destroy();

    return 0;
}
