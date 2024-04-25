cmake_minimum_required(VERSION 3.16)

# Project
project(CrisOptix CXX CUDA)
#set(CMAKE_CUDA_ARCHITECTURES 86)
set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
message("OptiX Dir: " ${OptiX_INSTALL_DIR})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

file(GLOB_RECURSE PROJECT_INCLUDE "include/*.hpp")
file(GLOB_RECURSE PROJECT_SOURCE "src/*.cpp")
file(GLOB_RECURSE PROJECT_CUDA_SOURCE "src/*.cu")

add_executable(${PROJECT_NAME} ${PROJECT_INCLUDE} ${PROJECT_SOURCE} ${PROJECT_CUDA_SOURCE})
target_include_directories(${PROJECT_NAME} PUBLIC 
	"include" 
	${CUDA_INC_PATH} 
	${CUDAToolkit_INCLUDE_DIRS} 
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${OptiX_INSTALL_DIR}/include
	${OPENGL_INCLUDE_DIRS})

# Compile
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_20)

# Dependencies

# OPENGL
find_package(OpenGL REQUIRED)

# CUDA
find_package(CUDAToolkit 12.0 REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

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
