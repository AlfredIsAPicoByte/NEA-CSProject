#include "timeClass.h"

void Time::update()
{
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<float> frameTime = now - lastFrame;
    
    if (FPSLimit && targetFPS > 0.0f)
    {
        float targetFrameTime = 1.0f / targetFPS;
        if (frameTime.count() < targetFrameTime)
        {
            std::this_thread::sleep_for(std::chrono::duration<float>(targetFrameTime - frameTime.count()));
            now = std::chrono::steady_clock::now();
            frameTime = now - lastFrame;
        }
    }

    deltaTime = std::clamp(frameTime.count(), 0.0001f, 0.1f); // Clamp to avoid spikes
    frameRate = 1.0f / deltaTime;
    lastFrame = now;
}