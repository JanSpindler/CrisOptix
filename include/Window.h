#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <OutputBuffer.h>
#include <glm/glm.hpp>

class Window
{
public:
	static void Init(const int width, const int height, const bool resizable, const std::string& title);
	static void Destroy();

	static void HandleIO();
	static void Display(OutputBuffer<glm::u8vec3>& outputBuffer);

	static bool IsClosed();
	static bool IsResized();
	static uint32_t GetWidth();
	static uint32_t GetHeight();

private:
	// Window
	static inline GLFWwindow* m_Handle = nullptr;
	static inline int m_Width = 0;
	static inline int m_Height = 0;
	static inline bool m_Resized = false;

	// Texture
	static inline GLuint m_RenderTex = 0;

	// Shader
	static inline GLuint m_VertexShader = 0;
	static inline GLuint m_FragmentShader = 0;
	static inline GLuint m_Program = 0;

	// Functions
	static void InitGlfw(const bool resizable, const std::string& title);
	static void InitOpenGl();
	static void InitGlsl();
};
