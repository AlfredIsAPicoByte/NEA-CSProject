#include "Engine.h"

void Engine::Start()
{
    state = State::STARTING;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    AppendMessage("Engine Started.");
}

void Engine::PausePlay()
{
    if (state == State::RUNNING) {
        state = State::PAUSED;
    } else if (state == State::PAUSED) {
        state = State::RUNNING;
    }
}

void Engine::Update(GLFWwindow* window, std::function<void()> processing, std::function<void()> render, std::function<void()> gui)
{
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    state = State::RUNNING;

    while (!glfwWindowShouldClose(window))
    {
        // Process user input
        if (!io.WantCaptureKeyboard && !io.WantCaptureMouse){
            processing();
            
            InputManager::Instance(window).doWhenKey(GLFW_KEY_ESCAPE, false, true, [&]()
            {
                glfwSetWindowShouldClose(window, true);
            });
        }

        if (state == State::PAUSED) {
            continue;
        }

        clearScreen();

        // Render the scene
        render();
        

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        gui();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void Engine::Exit()
{
    state = State::STOPPED;

	AppendMessage("Engine Stopped.");
}

void Engine::CleanUp(GLFWwindow* window)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Delete window before ending the program
    glfwDestroyWindow(window);
    // Terminate GLFW before ending the program
    glfwTerminate();

	AppendMessage("Destroyed window and terminated GLFW.");
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