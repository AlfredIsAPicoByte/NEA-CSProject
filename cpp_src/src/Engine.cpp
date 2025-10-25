#include "Engine.h"
#include "colorClass.h"
#include "Debug.h"

void Exit(GLFWwindow *window)
{
    glfwSetWindowShouldClose(window, true);
}
void WaitForEscape(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) Exit(window);
}

int InitGLFW()
{
    // Initialize GLFW
    if (!glfwInit()) {
		std::string msg = std::string("Failed to initialize GLFW");
		AppendOpenGLError(msg);
		throw std::runtime_error(msg);
	}

    // Tell GLFW what version of OpenGL we are using 
    // Example: OpenGL 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Enable debug context (optional)
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    return 0;
}

int InitGLAD()
{
    // Load GLAD to configure OpenGL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		AppendOpenGLError("Failed to initialize GLAD");
        return -1;
    }
    return 0;
}


GLFWwindow* createWindow(int width, int height, const char* title)
{
    // Create a GLFWwindow object of 800 by 800 pixels, naming it "YoutubeOpenGL"
    GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);
    // Error check if the window fails to create
    if (window == NULL)
    {
		AppendOpenGLError("Failed to create GLFW window");
        glfwTerminate();
        return nullptr;
    }

    // Specify the viewport of OpenGL in the Window
    // In this case the viewport goes from x = 0, y = 0, to x = width, y = height
    glViewport(0, 0, width, height);

    return window;
}

void MainLoop(GLFWwindow* window, std::function<void(GLFWwindow*)> renderFunc)
{
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

void applyClearColor(const Color& color) {
    glClearColor(color.r, color.g, color.b, color.a);
}