#pragma once

#include <iostream>
#include <GLFW/glfw3.h>
#include <thread>
#include <chrono>

class Time
{
public:
    float frameRate = 0.0f; 
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    bool FPSLimit = false;
    float targetFPS = 60.0f;

    // Call this once per frame (e.g., at the start of your main loop)
    void update();
};
