#pragma once
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Mesh.h"
#include "shaderClass.h"
#include "colorClass.h"
#include "Debug.h"

void Exit(GLFWwindow *window);
void WaitForEscape(GLFWwindow *window);

int InitGLFW();
int InitGLAD();
GLFWwindow* CreateWindow(int width, int height, const char* title);
void CleanUp(GLFWwindow* window);
