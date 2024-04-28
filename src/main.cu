#include <iostream>
#include <Window.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_host.h>
#include <cuda_gl_interop.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

OptixDeviceContext optixContext = nullptr;

void InitOptix()
{
    if (cudaFree(0) != CUDA_SUCCESS) { exit(1); }
    if (optixInit() != OPTIX_SUCCESS) { exit(1); }

    const OptixDeviceContextOptions optixContextOptions{};
    optixContext = nullptr;
    if (optixDeviceContextCreate(0, &optixContextOptions, &optixContext) != OPTIX_SUCCESS)
    {
        exit(1);
    }
}

int main()
{
    std::cout << "Hello there" << std::endl;

    Window::Init(800, 600, true, "CrisOptix");
    InitOptix();

    while (!Window::IsClosed())
    {
        Window::Update();
    }

    Window::Destroy();

    return 0;
}
