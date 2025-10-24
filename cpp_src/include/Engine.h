#pragma once
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>

#include "Debug.h"
#include "Mesh.h"
#include "colorClass.h"


void Exit(GLFWwindow *window);
void WaitForEscape(GLFWwindow *window);

int InitGLFW();
int InitGLAD();
GLFWwindow* createWindow(int width, int height, const char* title);
void MainLoop(GLFWwindow* window, std::function<void(GLFWwindow*)> renderFunc);
void CleanUp(GLFWwindow* window);
