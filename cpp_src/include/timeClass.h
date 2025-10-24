#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <sstream>

class Time
{
public:
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    bool FPSLimit = false;
    float targetFPS = 60.0f;

    // Call this once per frame (e.g., at the start of your main loop)
    void update();

    float displayFPS(GLFWwindow* window);
};

