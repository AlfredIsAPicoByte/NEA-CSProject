#include "Engine.h"

void Exit(GLFWwindow *window)
{
    glfwSetWindowShouldClose(window, true);
}
void WaitForEscape(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) Exit(window);
}

void CleanUp(GLFWwindow* window)
{
    // Delete window before ending the program
    glfwDestroyWindow(window);
    // Terminate GLFW before ending the program
    glfwTerminate();
}

void applyClearColor(const Color& color) {
    glClearColor(color.r, color.g, color.b, color.a);
}