#include <graph/Window.h>
#include <util/custom_assert.h>
#include <glsl.h>

static const std::string vertShaderSrc = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, -vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

static const std::string fragShaderSrc = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

void Window::Init(const int width, const int height, const bool resizable, const std::string& title)
{
	m_Width = width;
	m_Height = height;

	InitGlfw(resizable, title);
	InitOpenGl();
	InitRenderTex();
	InitVertexBuffer();
	InitProgram();
}

void Window::Destroy()
{
	glfwTerminate();
}

void Window::HandleIO()
{
	glfwPollEvents();

	const int oldWidth = m_Width;
	const int oldHeight = m_Height;
	glfwGetFramebufferSize(m_Handle, &m_Width, &m_Height);
	m_Resized = oldWidth != m_Width || oldHeight != m_Height;

	glfwGetCursorPos(m_Handle, &m_CursorX, &m_CursorY);
}

void Window::Display(const GLuint pbo)
{	
	// Framebuffer and viewport
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_Width, m_Height);

	// Clear
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use glsl program
	glUseProgram(m_Program);

	// Setupt render texutre
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_RenderTex);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, m_Width, m_Height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glUniform1i(m_RenderTexUniformLoc, 0);

	// Setup vertex array
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, m_VertexBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 6);

	//
	glDisableVertexAttribArray(0);

	CHECK_GL_ERROR();

	// Swap buffers
	glfwSwapBuffers(m_Handle);
}

bool Window::IsClosed()
{
	return glfwWindowShouldClose(m_Handle) == GLFW_TRUE;
}

bool Window::IsResized()
{
	return m_Resized;
}

uint32_t Window::GetWidth()
{
	return static_cast<uint32_t>(m_Width);
}

uint32_t Window::GetHeight()
{
	return static_cast<uint32_t>(m_Height);
}

bool Window::IsKeyPressed(const int keyCode)
{
	const int state = glfwGetKey(m_Handle, keyCode);
	return state == GLFW_PRESS || state == GLFW_REPEAT;
}

bool Window::IsMouseButtonPressed(const int button)
{
	const int state = glfwGetMouseButton(m_Handle, button);
	return state == GLFW_PRESS;
}

glm::vec2 Window::GetMousePos()
{
	return { m_CursorX, m_CursorY };
}

void Window::EnableCursor(const bool enableCursor)
{
	const int cursorMode = enableCursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED;
	glfwSetInputMode(m_Handle, GLFW_CURSOR, cursorMode);
}

void Window::InitGlfw(const bool resizable, const std::string& title)
{
	if (glfwInit() != GLFW_TRUE)
	{
		exit(1);
	}

	glfwWindowHint(GLFW_RESIZABLE, resizable ? GLFW_TRUE : GLFW_FALSE);
	m_Handle = glfwCreateWindow(m_Width, m_Height, title.c_str(), nullptr, nullptr);
	if (m_Handle == nullptr)
	{
		exit(1);
	}

	glfwMakeContextCurrent(m_Handle);
}

void Window::InitOpenGl()
{
	// Glad
	gladLoadGL();
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		exit(1);
	}

	// Viewport
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glViewport(0, 0, m_Width, m_Height);

	// Check gl errors
	CHECK_GL_ERROR();
}

void Window::InitRenderTex()
{
	glGenTextures(1, &m_RenderTex);
	glBindTexture(GL_TEXTURE_2D, m_RenderTex);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	CHECK_GL_ERROR();
}

void Window::InitVertexBuffer()
{
	static constexpr GLfloat vertexBufferData[] = {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f,
	};

	glGenBuffers(1, &m_VertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_VertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexBufferData), reinterpret_cast<const void*>(vertexBufferData), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	CHECK_GL_ERROR();
}

void Window::InitProgram()
{
	m_Program = CreateGlProgram(vertShaderSrc.c_str(), fragShaderSrc.c_str());
	m_RenderTexUniformLoc = GetGlUniformLoc(m_Program, "render_tex");

	CHECK_GL_ERROR();
}
