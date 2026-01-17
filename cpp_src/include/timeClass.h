#pragma once

#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>

struct Checkpoint
{
    std::chrono::steady_clock::time_point time;
    std::string data;

    float get_time_passed();
    bool wait_until_time_passed(std::chrono::steady_clock::duration duration);
};

class Time
{
public:
    float frameRate = 0.0f; 
    float deltaTime = 0.0f;
    std::chrono::steady_clock::time_point lastFrame = std::chrono::steady_clock::now();

    bool FPSLimit = false; // TODO: make frame limiting logic
    float targetFPS = 60.0f;

    // Call this once per frame (e.g., at the start of your main loop)
    void update();
};
