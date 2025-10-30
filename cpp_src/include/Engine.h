#pragma once
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>

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
    // Access the single global instance (Meyers' singleton, thread-safe since C++11)
    static Engine& Instance()
    {
        static Engine instance;
        return instance;
    }

    EngineState state = EngineState::STOPPED;

    void Start();
    void PausePlay();
    void Update(GLFWwindow* window, Time timer, std::function<void()> render);
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