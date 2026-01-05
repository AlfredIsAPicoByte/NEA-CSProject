#include "InputManager.h"

void InputManager::processInputs(const std::vector<ActionInput>& inputs, std::vector<bool> is_mice)
{
    for (const auto& [input, is_mouse] : zip(inputs, is_mice)) {
        if (!is_mouse) doWhenKey(input);
        if (is_mouse) doWhenMouseKey(input);
    }
}

void InputManager::doWhenKey(GLint key, bool isPressed, std::function<void()> action)
{
    // Safety check in case window is not initialized
    if (!m_window) return;

    int state = glfwGetKey(m_window, key);

    // If we want it pressed and it IS pressed, OR we want it released and it IS released
    if ((isPressed && state == GLFW_PRESS) || (!isPressed && state == GLFW_RELEASE)) {
        action();
    }
}

void InputManager::doWhenMouseKey(GLint key, bool isPressed, std::function<void()> action)
{
    if (!m_window) return;

    int state = glfwGetMouseButton(m_window, key);

    if ((isPressed && state == GLFW_PRESS) || (!isPressed && state == GLFW_RELEASE)) {
        action();
    }
}

void InputManager::doWhenKey(ActionInput input)
{
    doWhenKey(input.key, input.isPressed, input.action);
}

void InputManager::doWhenMouseKey(ActionInput input)
{
    doWhenMouseKey(input.key, input.isPressed, input.action);
}

void InputManager::getMousePosition(double& xpos, double& ypos)
{
    if (!m_window) return;
    glfwGetCursorPos(m_window, &xpos, &ypos);
}

void InputManager::setMousePosition(double xpos, double ypos)
{
    if (!m_window) return;
    glfwSetCursorPos(m_window, xpos, ypos);
}

void InputManager::setCursorVisibility(bool isVisible)
{
    if (!m_window) return;

    if (isVisible) {
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    } else {
        // CHANGED: Use DISABLED instead of HIDDEN.
        // DISABLED locks the mouse to the window allows infinite scrolling for 3D cameras.
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

void InputManager::toggleCursor()
{
    if (!m_window) return;

    int mode = glfwGetInputMode(m_window, GLFW_CURSOR);
    if (mode == GLFW_CURSOR_NORMAL) {
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    } else {
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}