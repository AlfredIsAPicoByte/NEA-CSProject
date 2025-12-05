#pragma once

#include <vector>
#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <functional>

#include "shaderClass.h"
#include "Model.h"
#include "textureClasse.h"
#include "cameraClass.h"
#include "IRenderable.h"
#include "PythonManager.h"

class Scene{
public:
    Scene();
    ~Scene();

    bool pythonRenderingUsed = false;

    int selectedMeshIndex = -1;

    void Initialize();

    void Render(std::function<void()> processing = nullptr, std::function<void()> renderStep = nullptr, std::function<void()> postProcessing = nullptr, std::function<void()> fallBack = nullptr);
    void UpdateScene();

    void LoadModel(const std::string& modelPath, Shader& shader);
    void SetCamera(Camera* camera);
    void SetOpenGLRenderFunction(std::shared_ptr<std::function<void()>> renderFunc);

    void CleanUp();
private:
    Camera* sceneCamera;

    std::vector<std::shared_ptr<IRenderable>> renderables;

    py::object pyAlgorithimModule;
    py::object pyRaytracingModule;
    py::object pySamplerModule;
    py::object pySceneModule;
    py::object pyCameraModule;
    py::object pyLuminanceModule;
    py::object pyGeometryModule;

    std::shared_ptr<std::function<void()>> openGLRenderFunction;
};