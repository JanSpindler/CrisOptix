#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <OutputBuffer.h>
#include <glm/glm.hpp>
#include <DeviceBuffer.h>

class Window
{
public:
	static void Init(const int width, const int height, const bool resizable, const std::string& title);
	static void Destroy();

	static void Update();
	static bool IsClosed();

private:
	static inline GLFWwindow* m_Handle = nullptr;
	static inline int m_Width = 0;
	static inline int m_Height = 0;

	static inline GLuint m_HdrTexture = 0;

	static inline GLuint m_VertexShader = 0;
	static inline GLuint m_FragmentShader = 0;
	static inline GLuint m_Program = 0;

	static void InitGlfw(const bool resizable, const std::string& title);
	static void InitOpenGl();
	static void InitGlsl();
};
