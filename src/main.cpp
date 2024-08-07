﻿#include <iostream>
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
#include <model/Scene.h>
#include <graph/SimpleRenderer.h>
#include <chrono>
#include <cmath>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <model/Emitter.h>

void MyOptixLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    Log::Info("OptiX Log: " + std::string(tag) + ": " + std::string(message));
}

static void InitCuda()
{
    std::array<int, 4> cudaGlDevices{};
    uint32_t glDeviceCount = 4;
    cudaGLGetDevices(&glDeviceCount, cudaGlDevices.data(), glDeviceCount, cudaGLDeviceListAll);
    if (glDeviceCount == 0) { Log::Error("No cuda gl capable device found"); }

    ASSERT_CUDA(cudaFree(nullptr));
}

static OptixDeviceContext InitOptix()
{
    ASSERT_OPTIX(optixInit());

    CUcontext cudaContext = nullptr;
    Log::Assert(cuCtxGetCurrent(&cudaContext) == CUDA_SUCCESS);

    OptixDeviceContextOptions options{};
    options.logCallbackFunction = MyOptixLogCallback;
    options.logCallbackData = nullptr;
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    OptixDeviceContext optixContext = nullptr;
    ASSERT_OPTIX(optixDeviceContextCreate(cudaContext, &options, &optixContext));

    return optixContext;
}

static void HandleCamMove(const float deltaTime, Camera& cam)
{
    bool changed = false;

    // Key movement
    const float camSpeed = Window::IsKeyPressed(GLFW_KEY_LEFT_SHIFT) ? 20.0f : 5.0f;
    glm::vec3 camMove(0.0f);

    if (Window::IsKeyPressed(GLFW_KEY_W)) { camMove += glm::vec3(0.0f, 0.0f, 1.0f); }
    if (Window::IsKeyPressed(GLFW_KEY_S)) { camMove += glm::vec3(0.0f, 0.0f, -1.0f); }
    if (Window::IsKeyPressed(GLFW_KEY_A)) { camMove += glm::vec3(-1.0f, 0.0f, 0.0f); }
    if (Window::IsKeyPressed(GLFW_KEY_D)) { camMove += glm::vec3(1.0f, 0.0f, 0.0f); }
    if (Window::IsKeyPressed(GLFW_KEY_SPACE)) { camMove += glm::vec3(0.0f, 1.0f, 0.0f); }
    if (Window::IsKeyPressed(GLFW_KEY_C)) { camMove += glm::vec3(0.0f, -1.0f, 0.0f); }

    const float camMoveLen = glm::length(camMove);
    if (camMoveLen > 0.001f)
    { 
        camMove = glm::normalize(camMove) * deltaTime * camSpeed;
        changed = true;
    }
    if (!std::isinf(camMoveLen) && !std::isnan(camMoveLen)) { cam.Move(camMove); }

    // Mouse rotation
    static glm::vec2 prevMousePos(0.0f);
    const glm::vec2 newMousePos = Window::GetMousePos();
    const glm::vec2 mouseMove = (newMousePos - prevMousePos) * -0.001f;
    prevMousePos = newMousePos;
    
    const bool rightMouseBtn = Window::IsMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT);
    Window::EnableCursor(!rightMouseBtn);
    if (rightMouseBtn)
    { 
        cam.RotateViewDir(mouseMove.x, mouseMove.y); 
        if (glm::length(mouseMove) > 0.001f) { changed = true; }
    }

    // Changed?
    cam.SetChanged(changed);
}

static void RenderAndDispay(
    OutputBuffer<glm::u8vec3>& outputBuffer, 
    SimpleRenderer& renderer, 
    DeviceBuffer<glm::vec3>& hdrBuffer,
    const uint32_t width,
    const uint32_t height)
{
    const size_t pixelCount = width * height;

    // Render
    outputBuffer.MapCuda();
    renderer.LaunchFrame(hdrBuffer.GetPtr());

    // Tonemapping
    CuBufferView<glm::vec3> hdrBufferView(hdrBuffer.GetCuPtr(), hdrBuffer.GetCount());
    CuBufferView<glm::u8vec3> ldrBufferView(outputBuffer.GetPixelDevicePtr(), pixelCount);
    ToneMapping(hdrBufferView, ldrBufferView);
    ASSERT_CUDA(cudaDeviceSynchronize());

    // Display
    outputBuffer.UnmapCuda();
    ASSERT_CUDA(cudaDeviceSynchronize());
    Window::Display(outputBuffer.GetPbo());
}

int main()
{
    std::cout << "Hello there" << std::endl;

    // Settings
    static constexpr uint32_t width = 800;
    static constexpr uint32_t height = 600;
    static constexpr size_t pixelCount = width * height;

    // Init
    Window::Init(width, height, false, "CrisOptix");
    InitCuda();
    const OptixDeviceContext optixDeviceContext = InitOptix();

    // Screen buffers
    OutputBuffer<glm::u8vec3> outputBuffer(width, height);
    DeviceBuffer<glm::vec3> hdrBuffer(pixelCount);

    // Camera
    Camera cam(
        glm::vec3(0.0f, 1.0f, 4.0f),
        glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        static_cast<float>(width) / static_cast<float>(height),
        glm::radians(60.0f));

    // Models
    //const Model cubeModel("./data/model/basic/cube.obj", false, SpecTexUsage::Color, optixDeviceContext);
    //const glm::mat4 cubeTransform = glm::translate(glm::scale(glm::identity<glm::mat4>(), glm::vec3(0.5f)), { 15.0f, 5.0f, 0.0f });
    //const ModelInstance cubeInstance(cubeModel, cubeTransform);
    //const Emitter cubeEmitter(cubeModel.GetMesh(0), cubeTransform, glm::vec3(10.0f));

    //const Model backpackModel("./data/model/backpack/backpack.obj", false, optixDeviceContext);
    //const ModelInstance backpackInstance(backpackModel, glm::mat4(1.0f));

    //const Model zeroDayModel("./data/model/ZeroDay_v1/MEASURE_SEVEN/MEASURE_SEVEN.fbx", false, SpecTexUsage::OccRoughMetal, optixDeviceContext);
    //const ModelInstance zeroDayInstance(zeroDayModel, glm::mat4(1.0f));

    const Model cornellModel("./data/model/Cornell/CornellBox-Original.obj", false, SpecTexUsage::Color, optixDeviceContext);
    const ModelInstance cornellInstance(cornellModel, glm::identity<glm::mat4>());

    //const Model bistroModel("./data/model/Bistro_v5_2/BistroInterior.fbx", true, SpecTexUsage::OccRoughMetal, optixDeviceContext);
    //const ModelInstance bistroInstance(bistroModel, glm::identity<glm::mat4>());

    // Scene
    const std::vector<const ModelInstance*> modelInstances = { &cornellInstance };
    //const std::vector<const ModelInstance*> modelInstances = { &bistroInstance };
    //const std::vector<const ModelInstance*> modelInstances = { &zeroDayInstance };
    const std::vector<const Emitter*> emitters = { };
    Scene scene(optixDeviceContext, modelInstances, emitters);

    // Renderer
    SimpleRenderer renderer(width, height, optixDeviceContext, cam, scene);

    // Main loop
    ASSERT_CUDA(cudaDeviceSynchronize());
    auto lastTime = std::chrono::high_resolution_clock::now();
    float totalTime = 0.0f;
    float renderTime = 0.0f;
    while (!Window::IsClosed())
    {
        // Delta time
        const auto newTime = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<float> duration = newTime - lastTime;
        const float deltaTime = duration.count();
        totalTime += deltaTime;
        lastTime = newTime;

        // Camera movement
        HandleCamMove(deltaTime, cam);

        // Handle window io // TODO: Handle resize
        Window::HandleIO();

        // Imgui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("General Info");

        ImGui::Text("Total Delta Time: %fs", deltaTime);
        ImGui::Text("Render Time: %fs", renderTime);
        ImGui::Text("Frame Index: %d", renderer.GetFrameIdx());

        const glm::vec3& camPos = cam.GetPos();
        ImGui::Text("Cam Pos: (%f, %f, %f)", camPos.x, camPos.y, camPos.z);

        const glm::vec3& camDir = cam.GetViewDir();
        ImGui::Text("Cam Dir: (%f, %f, %f)", camDir.x, camDir.y, camDir.z);

        renderer.RunImGui();

        ImGui::End();
        ImGui::Render();
            
        // Render and display
        const auto renderStart = std::chrono::high_resolution_clock::now();
        RenderAndDispay(outputBuffer, renderer, hdrBuffer, width, height);
        const auto renderEnd = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<float> renderDuration = renderEnd - renderStart;
        renderTime = renderDuration.count();
    }

    // End
    Window::Destroy();

    return 0;
}
