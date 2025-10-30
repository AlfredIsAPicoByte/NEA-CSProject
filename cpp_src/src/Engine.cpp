#include "Engine.h"

void Engine::Start()
{
    state = EngineState::STARTING;
}

void Engine::PausePlay()
{
    if (state == EngineState::RUNNING) {
        state = EngineState::PAUSED;
    } else if (state == EngineState::PAUSED) {
        state = EngineState::RUNNING;
    }
}

void Engine::Update(GLFWwindow* window, std::function<void()> render)
{
    state = EngineState::RUNNING;

    while (!glfwWindowShouldClose(window))
    {
        clearScreen();
        // Process user input
		awiatExitWindow(window);

        // Render the scene
        render();

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void Engine::Exit()
{
    state = EngineState::STOPPED;
	AppendMessage("Engine Stopped.");
}

void Engine::cleanUp(GLFWwindow* window)
{
    // Delete window before ending the program
    glfwDestroyWindow(window);
    // Terminate GLFW before ending the program
    glfwTerminate();

	AppendMessage("Destroyed window and terminated GLFW.");
    Exit();
}

void Engine::applyClearColor(const Color& color) {
    glClearColor(color.r, color.g, color.b, color.a);
}
void Engine::setDepthTest(bool enable) {
    if (enable) {
        glEnable(GL_DEPTH_TEST);
        AppendMessage("Depth testing enabled.");
    } else {
        glDisable(GL_DEPTH_TEST);
        AppendMessage("Depth testing disabled.");
    }
}

void Engine::clearScreen() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}