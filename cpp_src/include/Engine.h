#pragma once
#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include "Debug.h"
#include "colorClass.h"
#include "Mesh.h"

void Exit(GLFWwindow *window);
void WaitForEscape(GLFWwindow *window);

void InitGLFW();
void InitGLAD();
GLFWwindow* createWindow(int width, int height, const char* title);
void SetupOpenGLDebug();
void MainLoop(GLFWwindow* window, void (*renderFunction)(GLFWwindow* window), Color backgroundColor);
void CleanUp(GLFWwindow* window);
