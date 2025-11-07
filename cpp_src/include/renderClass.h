#pragma once

#include <string>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <functional>

#include "pythonManager.h"
#include "shaderClass.h"
#include "Model.h"
#include "textureClass.h"
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
    void loadTexture(const std::string& texturePath);
    void setCamera(Camera* cam);
private:
    Shader* shaderProgram;
    std::vector<Mesh*> meshes;
    std::vector<Texture*> textures;
    Camera* sceneCamera;

    py::object pyCamera;
    py::object pySampler;
    py::object pyScene;
    py::object pyBaseAlgorithm;
    py::object pyRayTracer;
}