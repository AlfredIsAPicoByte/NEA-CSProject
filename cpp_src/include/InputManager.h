#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Debugger.h"

struct ActionInput {
    GLint key;
    bool isPressed;
    std::function<void()> action;
};

class InputManager {
public:
    static InputManager& Instance(GLFWwindow* window)
    {
        static InputManager instance(window);
        if (window != instance.m_window) {
            instance.m_window = window;
        }
        return instance;
    }

    void processInputs(const std::vector<ActionInput>& inputs);
    
    // Polling wrappers
    void doWhenKey(GLint key, bool isPressed, std::function<void()> action);
    void doWhenKey(ActionInput input);
    
    void doWhenMouseKey(GLint key, bool isPressed, std::function<void()> action);
    void doWhenMouseKey(ActionInput input);

    // Mouse control
    void getMousePosition(double& xpos, double& ypos);
    void setMousePosition(double xpos, double ypos);
    void setCursorVisibility(bool isVisible);
    
    // Toggle helper
    void toggleCursor(); // Simplified

    // Delete copy constructors
    InputManager(const InputManager&) = delete;
    InputManager& operator=(const InputManager&) = delete;
    InputManager(InputManager&&) = delete;
    InputManager& operator=(InputManager&&) = delete;

private:
    explicit InputManager(GLFWwindow* window) : m_window(window) {}
    ~InputManager() = default;

    GLFWwindow* m_window;
};