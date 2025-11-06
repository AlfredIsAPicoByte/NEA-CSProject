#pragma once

#include <chrono>
#include <thread>
#include <algorithm>

class Time
{
public:
    float frameRate = 0.0f; 
    float deltaTime = 0.0f;
    std::chrono::steady_clock::time_point lastFrame = std::chrono::steady_clock::now();

    bool FPSLimit = false;
    float targetFPS = 60.0f;

    // Call this once per frame (e.g., at the start of your main loop)
    void update();
};
