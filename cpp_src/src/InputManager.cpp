#include "InputManager.h"

void InputManager::processInputs(std::vector<ActionInput> inputs)
{
    for (const auto& inp : inputs)
    {
        doWhenKey(inp);
    }
}

void InputManager::doWhenKey(GLint key, bool isPressed, std::function<void()> action)
{
    int state = glfwGetKey(m_window, key);

    if ((isPressed && state == GLFW_PRESS) || (!isPressed && state == GLFW_RELEASE)) {
        action();
    }
}

void InputManager::doWhenMouseKey(GLint key, bool isPressed, std::function<void()> action)
{
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
    glfwGetCursorPos(m_window, &xpos, &ypos);
}

void InputManager::setMousePosition(double xpos, double ypos)
{
    glfwSetCursorPos(m_window, xpos, ypos);
}

void InputManager::setCursorVisibility(bool isVisible)
{
    if (isVisible) {
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    } else {
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    }
}

void InputManager::toggleCursor(bool isEnabled)
{
    if (isEnabled) {
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    } else {
        glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}