#include "Engine.h"

void Exit(GLFWwindow *window)
{
    glfwSetWindowShouldClose(window, true);
}
void WaitForEscape(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) Exit(window);
}

void InitGLFW()
{
    // Initialize GLFW
    glfwInit();

    // Tell GLFW what version of OpenGL we are using 
    // In this case we are using OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // Tell GLFW we are using the CORE profile
    // So that means we only have the modern functions
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // enable error logging
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
}
void InitGLAD()
{
    //Load GLAD so it configures OpenGL
    gladLoadGL();
}

GLFWwindow* createWindow(int width, int height, const char* title)
{
    // Create a GLFWwindow object of 800 by 800 pixels, naming it "YoutubeOpenGL"
    GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);
    // Error check if the window fails to create
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    // Specify the viewport of OpenGL in the Window
    // In this case the viewport goes from x = 0, y = 0, to x = width, y = height
    glViewport(0, 0, width, height);

    return window;
}

void SetupOpenGLDebug()
{
    int flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        std::cout << "OpenGL debug callback" << std::endl;

        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); 
        glDebugMessageCallback(glDebugOutput, nullptr);glDebugMessageControl(GL_DEBUG_SOURCE_API, 
                      GL_DEBUG_TYPE_ERROR, 
                      GL_DEBUG_SEVERITY_HIGH,
                      0, nullptr, GL_TRUE); 
    } else {
        std::cout << "No debug context available" << std::endl;
    }
}

void MainLoop(GLFWwindow* window, std::function<void(GLFWwindow*)> renderFunc) {
    while (!glfwWindowShouldClose(window)) {
        renderFunc(window);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void CleanUp(GLFWwindow* window)
{
    // Delete window before ending the program
    glfwDestroyWindow(window);
    // Terminate GLFW before ending the program
    glfwTerminate();
}