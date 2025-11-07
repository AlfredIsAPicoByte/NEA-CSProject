#pragma once
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Mesh.h"
#include "shaderClass.h"
#include "colorClass.h"
#include "Debug.h"
#include "InputManager.h"
#include "pythonManager.h"

enum EngineState {
    STARTING,
    RUNNING,
    PAUSED,
    STOPPED
};

class Engine
{
public:
    static Engine& Instance()
    {
        static Engine instance;
        return instance;
    }

    EngineState state = EngineState::STOPPED;

    void Start();
    void PausePlay();
    void Update(GLFWwindow* window, std::function<void()> input, std::function<void()> render, std::function<void()> gui);
    void Exit();
    
    void cleanUp(GLFWwindow* window);
    void applyClearColor(const Color& color);
    void setDepthTest(bool enable);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;
private:
    Engine() = default;
    ~Engine() = default;
    
    void clearScreen();
};