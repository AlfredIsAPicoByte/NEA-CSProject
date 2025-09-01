#pragma once
#include <iostream>
#include <glad/glad.h>

// Check for OpenGL errors
GLenum glCheckError_(const char *file, int line);

// OpenGL debug callback
void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id, GLenum severity, GLsizei length, const char *message, const void *userParam);
