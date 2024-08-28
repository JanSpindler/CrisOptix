#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <graph/OutputBuffer.h>
#include <glm/glm.hpp>

class Window
{
public:
	static void Init(const int width, const int height, const bool resizable, const std::string& title);
	static void Destroy();

	static void HandleIO();
	static void Display(const GLuint pbo);

	static bool IsClosed();
	static bool IsResized();
	static uint32_t GetWidth();
	static uint32_t GetHeight();

	static bool IsKeyPressed(const int keyCode);
	static bool IsMouseButtonPressed(const int button);
	static glm::vec2 GetMousePos();

	static void EnableCursor(const bool enableCursor);

private:
	// Window
	static inline GLFWwindow* m_Handle = nullptr;
	static inline int m_Width = 0;
	static inline int m_Height = 0;
	static inline bool m_Resized = false;

	// Cursor
	static inline double m_CursorX = 0.0;
	static inline double m_CursorY = 0.0;

	// Texture
	static inline GLuint m_RenderTex = 0;
	static inline GLint m_RenderTexUniformLoc = -1;

	// Vertex buffer
	static inline GLuint m_VertexBuffer = 0;
	static inline GLuint m_VertexArray = 0;

	// Shader
	static inline GLuint m_VertexShader = 0;
	static inline GLuint m_FragmentShader = 0;
	static inline GLuint m_Program = 0;

	// Functions
	static void InitGlfw(const bool resizable, const std::string& title);
	static void InitOpenGl();
	static void InitRenderTex();
	static void InitVertexBuffer();
	static void InitProgram();
	static void InitImGui();
};
