#pragma once

#include <string>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>

#include "pythonManager.h"
#include "shaderClass.h"
#include "Model.h"
#include "textureClass.h"
#include "cameraClass.h"
#include "IRenderable.h"

class Scene{
public:
    Scene();
    ~Scene();

    bool pythonRenderingUsed = false;

    int selectedMeshIndex = -1;

    void initialize();
    void render();
    void update();
    void cleanup();
    void loadModel(const std::string& modelPath);
    void setCamera(Camera* cam);
private:
    Shader* shaderProgram;
    Camera* sceneCamera;

    std::vector<std::shared_ptr<IRenderable>> renderables;

    py::object pyCamera;
    py::object pySampler;
    py::object pyScene;
    py::object pyBaseAlgorithm;
    py::object pyRayTracer;
};