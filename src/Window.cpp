#include <Window.h>
#include <custom_assert.h>

static const std::string vertShaderSrx = R"(
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
	InitGlsl();
}

void Window::Destroy()
{
	glfwTerminate();
}

void Window::HandleIO()
{
	glfwPollEvents();

	int oldWidth = m_Width;
	int oldHeight = m_Height;
	glfwGetFramebufferSize(m_Handle, &m_Width, &m_Height);
	m_Resized = oldWidth != m_Width || oldHeight != m_Height;
}

void Window::Display(OutputBuffer<glm::u8vec3>& outputBuffer)
{
	//
	outputBuffer.MapCuda();

	outputBuffer.UnmapCuda();
	
	// Display
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_Width, m_Height);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(m_Program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_RenderTex);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, outputBuffer.GetPbo());

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

void Window::InitGlsl()
{
	static const std::string vsSource =
		"#version 330\n"
		"layout(location = 0) in vec4 attrPosition;\n"
		"layout(location = 8) in vec2 attrTexCoord0;\n"
		"out vec2 varTexCoord0;\n"
		"void main()\n"
		"{\n"
		"  gl_Position  = attrPosition;\n"
		"  varTexCoord0 = attrTexCoord0;\n"
		"}\n";

	static const std::string fsSource =
		"#version 330\n"
		"uniform sampler2D samplerHDR;\n"
		"in vec2 varTexCoord0;\n"
		"layout(location = 0, index = 0) out vec4 outColor;\n"
		"void main()\n"
		"{\n"
		"  outColor = texture(samplerHDR, varTexCoord0);\n"
		"}\n";

	GLint vsCompiled = 0;
	GLint fsCompiled = 0;

	m_VertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (m_VertexShader)
	{
		GLsizei len = (GLsizei)vsSource.size();
		const GLchar* vs = vsSource.c_str();
		glShaderSource(m_VertexShader, 1, &vs, &len);
		glCompileShader(m_VertexShader);
		//checkInfoLog(vs, m_VertexShader);

		glGetShaderiv(m_VertexShader, GL_COMPILE_STATUS, &vsCompiled);
		if (vsCompiled == GL_FALSE) { exit(1); }
	}

	m_FragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (m_FragmentShader)
	{
		GLsizei len = (GLsizei)fsSource.size();
		const GLchar* fs = fsSource.c_str();
		glShaderSource(m_FragmentShader, 1, &fs, &len);
		glCompileShader(m_FragmentShader);
		//checkInfoLog(fs, m_FragmentShader);

		glGetShaderiv(m_FragmentShader, GL_COMPILE_STATUS, &fsCompiled);
		if (fsCompiled == GL_FALSE) { exit(1); }
	}

	m_Program = glCreateProgram();
	if (m_Program)
	{
		GLint programLinked = 0;

		if (m_VertexShader && vsCompiled)
		{
			glAttachShader(m_Program, m_VertexShader);
		}
		if (m_FragmentShader&& fsCompiled)
		{
			glAttachShader(m_Program, m_FragmentShader);
		}

		glLinkProgram(m_Program);
		//checkInfoLog("m_glslProgram", m_glslProgram);

		glGetProgramiv(m_Program, GL_LINK_STATUS, &programLinked);
		if (programLinked == GL_FALSE) { exit(1); }

		if (programLinked)
		{
			glUseProgram(m_Program);
			glUniform1i(glGetUniformLocation(m_Program, "samplerHDR"), 0); // texture image unit 0
			glUseProgram(0);
		}
	}

	// Check gl errors
	CHECK_GL_ERROR();
}
