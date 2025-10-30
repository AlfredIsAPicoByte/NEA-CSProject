#include "timeClass.h"

void Time::update()
{
    double currentFrame = glfwGetTime();
    double frameTime = static_cast<double>(deltaTime);

    if (FPSLimit && targetFPS > 0.0f)
    {
        const double targetFrameTime = 1.0 / static_cast<double>(targetFPS);

        if (frameTime < targetFrameTime)
        {
            double sleepTime = targetFrameTime - frameTime;

            // Sleep for most of the remaining time (coarse), then busy-wait for finer accuracy.
            if (sleepTime > 0.003) // if more than ~3ms remaining, sleep most of it
                std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime - 0.001));

            // busy-wait until target frame time elapsed for better precision
            while ((glfwGetTime() - static_cast<double>(lastFrame)) < targetFrameTime) { /* spin */ }

            currentFrame = glfwGetTime();
            frameTime = static_cast<double>(currentFrame - lastFrame);
        }
    }
    
    deltaTime = static_cast<float>(currentFrame - lastFrame);
    if (deltaTime <= 0.0f) deltaTime = 1.0f / 1000.0f;
    lastFrame = static_cast<float>(currentFrame);

    // Smooth the reported frame time to reduce jitter (exponential moving average).
    static float smoothDelta = 0.0f;
    if (smoothDelta <= 0.0f) smoothDelta = deltaTime;
    const float alpha = 0.1f; // smaller value = smoother and more laggy
    smoothDelta = smoothDelta * (1.0f - alpha) + deltaTime * alpha;

    frameRate = 1.0f / smoothDelta;
}