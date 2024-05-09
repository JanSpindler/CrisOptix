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
#include <CuBufferView.h>
#include <kernel/tonemap.h>

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

    static constexpr size_t width = 800;
    static constexpr size_t height = 600;
    static constexpr size_t pixelCount = width * height;

    Window::Init(width, height, false, "CrisOptix");
    InitCuda();
    InitOptix();

    OutputBuffer<glm::u8vec3> outputBuffer(width, height);

    DeviceBuffer<glm::vec3> hdrBuffer(pixelCount);

    while (!Window::IsClosed())
    {
        // Handle window io
        Window::HandleIO();
        // TODO: handle resize by resizing buffers

        //
        outputBuffer.MapCuda();

        // TODO: trace rays

        // Tone mapping
        CuBufferView<glm::vec3> hdrBufferView(hdrBuffer.GetCuPtr(), hdrBuffer.GetCount());
        CuBufferView<glm::u8vec3> ldrBufferView(outputBuffer.GetPixelDevicePtr(), pixelCount);
        ToneMapping(hdrBufferView, ldrBufferView);
        ASSERT_CUDA(cudaDeviceSynchronize());

        //
        outputBuffer.UnmapCuda();
        ASSERT_CUDA(cudaDeviceSynchronize());

        // Render to window
        Window::Display(outputBuffer.GetPbo());
    }

    Window::Destroy();

    return 0;
}
