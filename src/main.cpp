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
#include <graph/Pipeline.h>
#include <graph/ShaderBindingTable.h>
#include <graph/Scene.h>

void MyOptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    Log::Info("OptiX Log: " + std::string(tag) + ": " + std::string(message));
}

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

    CUcontext cudaContext = nullptr;
    Log::Assert(cuCtxGetCurrent(&cudaContext) == CUDA_SUCCESS);

    OptixDeviceContext optixContext = nullptr;
    ASSERT_OPTIX(optixDeviceContextCreate(cudaContext, nullptr, &optixContext));
    ASSERT_OPTIX(optixDeviceContextSetLogCallback(optixContext, MyOptixLogCallback, nullptr, 4));

    return optixContext;
}

int main()
{
    std::cout << "Hello there" << std::endl;

    static constexpr size_t width = 800;
    static constexpr size_t height = 600;
    static constexpr size_t pixelCount = width * height;

    Window::Init(width, height, false, "CrisOptix");
    InitCuda();
    const OptixDeviceContext optixDeviceContext = InitOptix();

    OutputBuffer<glm::u8vec3> outputBuffer(width, height);

    DeviceBuffer<glm::vec3> hdrBuffer(pixelCount);

    ShaderEntryPointDesc raygenEntry{};
    raygenEntry.shaderKind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenEntry.fileName = std::string("test.ptx");
    raygenEntry.entryPointName = std::string("__raygen__main");

    ShaderEntryPointDesc missEntry{};
    missEntry.shaderKind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missEntry.fileName = "test.ptx";
    missEntry.entryPointName = "__miss__main";

    ShaderEntryPointDesc occlusionMissEntry{};
    occlusionMissEntry.shaderKind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    occlusionMissEntry.fileName = "test.ptx";
    occlusionMissEntry.entryPointName = "__miss__occlusion";

    const std::vector<ShaderEntryPointDesc> shaders = { raygenEntry, missEntry, occlusionMissEntry };
    Pipeline pipeline(optixDeviceContext, shaders);

    ShaderBindingTable sbt(
        optixDeviceContext,
        pipeline.GetRaygenProgramGroups(),
        pipeline.GetMissProgramGroups(),
        pipeline.GetExceptionProgramGroups(),
        pipeline.GetCallableProgramGroups(),
        pipeline.GetHitgroupProgramGroups());

    const Model dragonModel("./data/model/basic/dragon.obj", false, optixDeviceContext);
    const ModelInstance dragonInstance(dragonModel, glm::mat4(1.0f));

    const std::vector<ModelInstance> modelInstances = { dragonInstance };
    Scene scene(optixDeviceContext, modelInstances, pipeline, sbt);

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
