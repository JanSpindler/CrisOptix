cmake_minimum_required(VERSION 3.16)

# Project
project(CrisOptix CXX CUDA)

# CUDA and OPTIX
set(CMAKE_CUDA_ARCHITECTURES 86)
#option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)
#mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)
#set(CUDA_NVCC_FLAGS "--use_fast_math" "--expt-relaxed-constexpr" "-Wno-deprecated-gpu-targets" "-Xcudafe=\"--diag_suppress=2886\"")
set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
message("OptiX install dir: " ${OptiX_INSTALL_DIR})

# Module path
list(APPEND CMAKE_MODULE_PATH "./cmake")

# Other
#option(BUILD_SHARED_LIBS "Build shared libraries" ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

# Multiple processors on windows
if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

# Project files
file(GLOB_RECURSE PROJECT_INCLUDE "include/*.h")
file(GLOB_RECURSE PROJECT_SOURCE "src/*.cpp")
file(GLOB_RECURSE PROJECT_CUDA_SOURCE "src/*.cu")

add_executable(${PROJECT_NAME} ${PROJECT_INCLUDE} ${PROJECT_SOURCE} ${PROJECT_CUDA_SOURCE})
target_include_directories(${PROJECT_NAME} PUBLIC 
	"include" 
	${CUDA_INC_PATH} 
	${CUDAToolkit_INCLUDE_DIRS} 
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	"${OptiX_INSTALL_DIR}/include"
	${OPENGL_INCLUDE_DIRS})

# Compile
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_17)

# Dependencies

# OPENGL
find_package(OpenGL REQUIRED)

# CUDA
find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver CUDA::cudart ${CUDA_LIBRARIES})

# OPTIX
find_package(OptiX REQUIRED)

# GLAD
find_package(glad CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glad::glad)

# GLFW
find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

# GLM
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

# IMGUI
find_package(imgui CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE imgui::imgui)

# ASSIMP
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)
