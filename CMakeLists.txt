cmake_minimum_required(VERSION 3.16)

# Project
project(CrisOptix CXX CUDA)
message("CMAKE_CURRENT_LIST_DIR is ${CMAKE_CURRENT_LIST_DIR}")

# CUDA and OPTIX
set(CMAKE_CUDA_ARCHITECTURES 61)
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)
set(CUDA_NVCC_FLAGS "--use_fast_math" "--expt-relaxed-constexpr" "-Wno-deprecated-gpu-targets" "-Xcudafe=\"--diag_suppress=2886\"")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)

set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
message("OptiX install dir: " ${OptiX_INSTALL_DIR})

# Module path
list(APPEND CMAKE_MODULE_PATH "./cmake")

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include dirs
include_directories(${PROJECT_NAME} PUBLIC 
	"include" 
	${CUDA_INC_PATH} 
	${CUDAToolkit_INCLUDE_DIRS} 
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	"${OptiX_INSTALL_DIR}/include"
	${OPENGL_INCLUDE_DIRS})

# Glob
file(GLOB_RECURSE PROJECT_INCLUDE "include/*.h")
file(GLOB_RECURSE PROJECT_SOURCE "src/*.cpp")
file(GLOB_RECURSE PROJECT_CUDA_SOURCE "src/*.cu")
file(GLOB_RECURSE PROJECT_SHADER_SOURCE "shader/*.cu")

# Shader lib
add_library(CrisOptixShader OBJECT ${PROJECT_INCLUDE} ${PROJECT_SHADER_SOURCE})
set_target_properties(CrisOptixShader PROPERTIES 
	CUDA_PTX_COMPILATION ON
	CUDA_SEPARABLE_COMPILATION ON)

# Project files
add_executable(${PROJECT_NAME} ${PROJECT_INCLUDE} ${PROJECT_SOURCE} ${PROJECT_CUDA_SOURCE})
add_dependencies(${PROJECT_NAME} CrisOptixShader)

# Dependencies

# OPENGL
find_package(OpenGL REQUIRED)

# CUDA
find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cuda_driver CUDA::cudart ${CUDA_LIBRARIES})
target_link_libraries(CrisOptixShader PRIVATE CUDA::cuda_driver CUDA::cudart ${CUDA_LIBRARIES})

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
target_link_libraries(CrisOptixShader PRIVATE glm::glm)

# IMGUI
find_package(imgui CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE imgui::imgui)

# ASSIMP
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)

# STB
find_package(Stb REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})
