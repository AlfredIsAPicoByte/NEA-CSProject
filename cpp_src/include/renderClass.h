#pragma once

#include <string>
#include <vector>
#include <functional>

#include "pythonManager.h"
#include "shaderClass.h"
#include "Model.h"
#include "Camera.h"

class Scene{
public:
    Scene();
    ~Scene();

    int selectedMeshIndex = -1;

    void initialize();
    void render();
    void update();
    void cleanup();
    void loadModel(const std::string& modelPath);
    void setCamera(Camera* cam);
private:
    Shader* shaderProgram;
    std::vector<Model*> models;
    Camera* sceneCamera;

    py::object pyCamera;
    py::object pySampler;
    py::object pyScene;
    py::object pyBaseAlgorithm;
    py::object pyRayTracer;
};