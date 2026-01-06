#pragma once

#include <vector>
#include <string>
#include <functional>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include "Model.h"
#include "Image.hpp"
#include "IVirtualObject.hpp"
#include "ModelMeshAdapter.h"
#include "PythonManager.h"

struct RenderSettings {
    int imageWidth = 800;
    int imageHeight = 800;
    bool usePythonRendering = true;

    RenderSettings() = default;
    RenderSettings(int width = 800, int height = 800, bool pythonRendering = true)
        : imageWidth(width), imageHeight(height), usePythonRendering(pythonRendering) {}

    void SerializeFields(json& j) const {
        j["image_width"] = imageWidth;
        j["image_height"] = imageHeight;
        j["use_python_rendering"] = usePythonRendering;
    }
};

class Scene : public IVirtualObject {
public:
    Scene();
    ~Scene();

    std::string name = "Scene";

    Camera* sceneCamera;
    std::vector<std::shared_ptr<IVirtualObject>> sceneObjects;
    std::vector<std::shared_ptr<IRenderable>> renderables;
    int selectedRenderable = -1;
    
    std::shared_ptr<std::function<void()>> openGLRenderFunction;
    std::shared_ptr<RenderSettings> renderSettings;

    void Initialize();

    void Render(std::function<void()> preProcessing = nullptr, std::function<Image()> renderStep = nullptr, std::function<void()> postProcessing = nullptr, std::function<void()> fallBack = nullptr);
    void UpdateScene();
    bool SaveScene(const std::string& filePath);
    bool LoadScene(const std::string& filePath);
    bool SaveRenderedImage(const std::string& filePath);

    void SerializeFields(json& j) const override;
    void DeserializeFields(const json& j) override;

    void LoadModel(const std::string& modelPath, Shader& shader);
    void SetCamera(Camera* camera);
    void AddRenderable(std::shared_ptr<IRenderable> renderable);
    std::shared_ptr<IRenderable>  GetRenderable(const std::string& name);
    std::shared_ptr<IRenderable>  GetRenderable(int index);
    void RemoveRenderable(const std::string& name);
    void RemoveRenderable(int index);
    void ClearRenderables();
    void AddObject(std::shared_ptr<IVirtualObject> object);
    std::shared_ptr<IVirtualObject>  GetObject(const std::string& name);
    std::shared_ptr<IVirtualObject>  GetObject(int index);
    void RemoveObject(const std::string& name);
    void RemoveObject(int index);
    void ClearObjects();
    
    void SetOpenGLRenderFunction(std::shared_ptr<std::function<void()>> renderFunc);
    
    py::object pyAlgorithmModule;
    py::object pyRaytracingModule;
    py::object pySamplerModule;
    py::object pySceneModule;
    py::object pyCameraModule;
    py::object pyLuminanceModule;
    py::object pyGeometryModule;

    void CleanUp();
};