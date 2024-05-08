#include <Window.h>
#include <custom_assert.h>

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

void Window::Update()
{
	// GLFW
	glfwPollEvents();
	glfwGetFramebufferSize(m_Handle, &m_Width, &m_Height);

	// Render with OptiX
	// TODO

	// Check gl errors
	CHECK_GL_ERROR();
}

bool Window::IsClosed()
{
	return glfwWindowShouldClose(m_Handle) == GLFW_TRUE;
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
