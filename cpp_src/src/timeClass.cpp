#include "timeClass.h"

void Time::update()
{
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    if (FPSLimit)
    {
        float targetFrameTime = 1.0f / targetFPS;
        if (deltaTime < targetFrameTime)
        {
            float sleepTime = targetFrameTime - deltaTime;
            glfwWaitEventsTimeout(sleepTime);
            currentFrame = static_cast<float>(glfwGetTime());
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
        }
    }
}