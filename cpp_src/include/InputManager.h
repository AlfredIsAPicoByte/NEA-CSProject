#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Debug.h"

class InputManager {
public:
    static InputManager& Instance()
    {
        static InputManager instance;
        return instance;
    }

    InputManager(GLFWwindow* window);
    ~InputManager();

    void processInput();
    bool isKeyPressed(int key);
    bool isMouseButtonPressed(int button);
    void getMousePosition(double& xpos, double& ypos);
    void setMousePosition(double& xpos, double& ypos);
    void setCursorMode(int mode);
    void setScrollCallback(GLFWscrollfun callback);

    InputManager(const InputManager&) = delete;
    InputManager& operator=(const InputManager&) = delete;
    InputManager(InputManager&&) = delete;
    InputManager& operator=(InputManager&&) = delete;
private:
    InputManager() = default;
    ~InputManager() = default;

    GLFWwindow* m_window;
};