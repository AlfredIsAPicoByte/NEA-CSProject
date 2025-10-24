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

float Time::displayFPS(GLFWwindow* window)
{
    static float fpsTimer = 0.0f;
    static int frameCount = 0;
    frameCount++;
    fpsTimer += deltaTime;

    if (fpsTimer >= 1.0f)
    {
        float fps = frameCount / fpsTimer;
        std::ostringstream ss;
        ss << "FPS: " << fps;
        std::string title = ss.str();
        glfwSetWindowTitle(window, title.c_str());
        frameCount = 0;
        fpsTimer = 0.0f;
        return fps;
        fpsTimer = 0.0f;
        return fps;
    }
    
    return -1.0f; // Indicate that FPS was not updated this frame
}