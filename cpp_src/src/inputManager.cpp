#include "InputManager.h"

void awiatExitWindow(GLFWwindow* window)
{
    // Close window on pressing ESC
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}