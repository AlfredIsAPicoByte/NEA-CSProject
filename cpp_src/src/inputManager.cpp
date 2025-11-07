#include "InputManager.h"

InputManager::InputManager(GLFWwindow* window)
    : m_window(window)
{
    AppendMessage("InputManager initialized.")
}

InputManager::~InputManager()
{
    AppendMessage("InputManager destroyed.")
}

void InputManager::processInput()
{
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(m_window, true);
}

void InputManager::setCursorMode(int mode) // e.g., GLFW_CURSOR_NORMAL, GLFW_CURSOR_HIDDEN, GLFW_CURSOR_DISABLED
{
    glfwSetInputMode(m_window, GLFW_CURSOR, mode);
}

bool InputManager::isKeyPressed(int key) // e.g., GLFW_KEY_W
{
    return glfwGetKey(m_window, key) == GLFW_PRESS;
}

bool InputManager::isMouseButtonPressed(int button) // e.g., GLFW_MOUSE_BUTTON_LEFT
{
    return glfwGetMouseButton(m_window, button) == GLFW_PRESS;
}

void InputManager::getMousePosition(double& xpos, double& ypos)
{
    glfwGetCursorPos(m_window, &xpos, &ypos);
}

void InputManager::setMousePosition(double xpos, double ypos)
{
    glfwSetCursorPos(m_window, xpos, ypos);
}

void InputManager::setScrollCallback(GLFWscrollfun callback)
{
    glfwSetScrollCallback(m_window, callback);
}

