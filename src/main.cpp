#include <iostream>
#include <graph/Window.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_host.h>
#include <cuda_gl_interop.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <util/custom_assert.h>
#include <glm/glm.hpp>
#include <graph/OutputBuffer.h>
#include <graph/DeviceBuffer.h>
#include <array>
#include <graph/CuBufferView.h>
#include <kernel/tonemap.h>
#include <model/Model.h>

void InitCuda()
{
    std::array<int, 4> cudaGlDevices{};
    uint32_t glDeviceCount = 4;
    cudaGLGetDevices(&glDeviceCount, cudaGlDevices.data(), glDeviceCount, cudaGLDeviceListAll);
    if (glDeviceCount == 0) { Log::Error("No cuda gl capable device found"); }

    ASSERT_CUDA(cudaFree(nullptr));
}

OptixDeviceContext InitOptix()
{
    ASSERT_OPTIX(optixInit());

    OptixDeviceContext optixContext = nullptr;
    const OptixDeviceContextOptions optixContextOptions{};
    ASSERT_OPTIX(optixDeviceContextCreate(0, &optixContextOptions, &optixContext));

    return optixContext;
}

void TestModelLoading()
{
    Model zeroDayModel("./data/ZeroDay_v1/MEASURE_SEVEN/MEASURE_SEVEN.fbx", false);
}

int main()
{
    std::cout << "Hello there" << std::endl;

    static constexpr size_t width = 800;
    static constexpr size_t height = 600;
    static constexpr size_t pixelCount = width * height;

    Window::Init(width, height, false, "CrisOptix");
    InitCuda();
    OptixDeviceContext optixContext = InitOptix();

    OutputBuffer<glm::u8vec3> outputBuffer(width, height);

    DeviceBuffer<glm::vec3> hdrBuffer(pixelCount);

    TestModelLoading();

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
