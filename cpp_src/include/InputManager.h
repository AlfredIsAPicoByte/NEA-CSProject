#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Debug.h"

struct ActionInput {
    GLuint key;
    bool isMouse;
    bool isPressed;
    std::function<void()> action;
};

class InputManager {
public:
    static InputManager& Instance(GLFWwindow* window)
    {
        static InputManager instance(window);
        return instance;
    }

    void processInputs(std::vector<ActionInput> inputs);
    void doWhenKey(GLint key, bool isMouse, bool isPressed, std::function<void()> action);
    void doWhenKey(ActionInput input);
    void getMousePosition(double& xpos, double& ypos);
    void setMousePosition(double xpos, double ypos);
    void setCursorVisibility(bool isVisible);
    void toggleCursor(bool isEnabled);

    InputManager(const InputManager&) = delete;
    InputManager& operator=(const InputManager&) = delete;
    InputManager(InputManager&&) = delete;
    InputManager& operator=(InputManager&&) = delete;
private:
    InputManager() = delete;
    explicit InputManager(GLFWwindow* window) : m_window(window) {}
    ~InputManager() = default;

    GLFWwindow* m_window;
};