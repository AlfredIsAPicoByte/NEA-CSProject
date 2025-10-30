#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <glad/glad.h>

const long MAX_LOG_SIZE = 1048576; // 1 MB
extern std::string debugLog;

// Check for OpenGL errors
GLenum glCheckError_(const char *file, int line);

// OpenGL debug callback
void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id, GLenum severity, GLsizei length, const char *message, const void *userParam);
void APIENTRY EnableOpenGLDebugger();

#define GL_CHECK(x) do { \
    x; \
    glCheckError_(__FILE__, __LINE__); \
} while (0)

#if DEBUG_MODE
    #define GLCall(x) GL_CHECK(x)
#else
    #define GLCall(x) x
#endif

void AppendMessage(const std::string& msg);
void AppendWarning(const std::string& warning);
void AppendError(const std::string& error);
void AppendOpenGLMessage(const std::string& msg);
void AppendOpenGLWarning(const std::string& warning);
void AppendOpenGLError(const std::string& error);
void AppendPythonMessage(const std::string& msg);
void AppendPythonWarning(const std::string& warning);
void AppendPythonError(const std::string& error);
void PrintDebugLog();
void ClearDebugLog();