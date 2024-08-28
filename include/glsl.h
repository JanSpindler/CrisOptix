#pragma once

#include <GL/glew.h>
#include <string>

GLuint CreateGlShader(const char* src, const GLuint type);
GLuint CreateGlProgram(const char* vertSrc, const char* fragSrc);
GLint GetGlUniformLoc(const GLuint program, const std::string& name);
