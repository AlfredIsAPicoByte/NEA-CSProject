#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Mesh.h"
#include "Model.h"
#include "shaderClass.h"
#include "colorClass.h"
#include "cameraClass.h"
#include "Light.h"
#include "Debugger.h"
#include "InputManager.h"
#include "PythonManager.h"

class Engine
{
public:
    static Engine& Instance()
    {
        static Engine instance;
        return instance;
    }

    enum State {
        STARTING,
        RUNNING,
        PAUSED,
        STOPPED
    } state = State::STOPPED;

    void Start();
    void PausePlay();
    void Update(GLFWwindow* window, std::function<void()> input, std::function<void()> render, std::function<void()> gui);
    void Exit();
    
    void applyClearColor(const Color& color);
    void setDepthTest(bool enable);
    
    void CleanUp(GLFWwindow* window);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;
private:
    Engine() = default;
    ~Engine() = default;
    
    void clearScreen();
};