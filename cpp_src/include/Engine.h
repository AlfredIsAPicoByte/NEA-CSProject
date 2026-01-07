#pragma once

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>
#include <utility>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "renderClass.h"
#include "colorClass.h"
#include "ModelAdapter.h"
#include "Light.hpp"
#include "Material.hpp"
#include "InputManager.h"

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
    void Update(GLFWwindow* window, std::function<void()> preProcessing = nullptr, std::function<void()> input = nullptr, std::function<void()> renderStep = nullptr,  std::function<void()> postProcessing = nullptr, std::function<void()> gui = nullptr, std::function<void()> fallBack = nullptr);
    void Exit();
    
    void applyClearColor(const Color& color);
    void setDepthTest(bool enable);
    
    void CleanUp(GLFWwindow* window);
    
    static void FramebufferSizeCallback(GLFWwindow* window, int w, int h) {
        auto* engine = static_cast<Engine*>(glfwGetWindowUserPointer(window));
        engine->OnFramebufferResize(w, h);
    }
    void OnFramebufferResize(int w, int h) {
        glViewport(0, 0, w, h);
    }
    
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;
private:
    Engine() = default;
    ~Engine() = default;
    
    void clearScreen();
};