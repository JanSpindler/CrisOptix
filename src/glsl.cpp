#include <glsl.h>
#include <util/Log.h>
#include <util/custom_assert.h>

GLuint CreateGlShader(const char* src, const GLuint type)
{
	const GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);

	GLint compiled = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if (compiled == GL_FALSE)
	{
		GLint maxLen = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLen);

		std::string infoLog(maxLen, '\0');
		glGetShaderInfoLog(shader, maxLen, nullptr, infoLog.data());

		glDeleteShader(shader);

		Log::Error(infoLog);
	}

	CHECK_GL_ERROR();

	return shader;
}

GLuint CreateGlProgram(const char* vertSrc, const char* fragSrc)
{
	const GLuint vertShader = CreateGlShader(vertSrc, GL_VERTEX_SHADER);
	const GLuint fragShader = CreateGlShader(fragSrc, GL_FRAGMENT_SHADER);

	const GLuint program = glCreateProgram();
	glAttachShader(program, vertShader);
	glAttachShader(program, fragShader);
	glLinkProgram(program);

	GLint linked = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (linked == GL_FALSE)
	{
		GLint maxLen = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLen);

		std::string infoLog(maxLen, '\0');
		glGetProgramInfoLog(program, maxLen, nullptr, infoLog.data());

		glDeleteProgram(program);
		glDeleteShader(vertShader);
		glDeleteShader(fragShader);

		Log::Error(infoLog);
	}

	glDetachShader(program, vertShader);
	glDetachShader(program, fragShader);

	CHECK_GL_ERROR();

	return program;
}

GLint GetGlUniformLoc(const GLuint program, const std::string& name)
{
	const GLint loc = glGetUniformLocation(program, name.c_str());
	if (loc < 0)
	{
		Log::Error("Failed to get uniform location for " + name);
	}
	return loc;
}
