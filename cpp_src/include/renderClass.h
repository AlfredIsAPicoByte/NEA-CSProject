#pragma once

#include <vector>
#include <string>
#include <functional>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include "Model.h"
#include "Image.h"
#include "IVirtualObject.h"
#include "PythonManager.h"

struct RenderSettings {
    int imageWidth = 800;
    int imageHeight = 800;
    int samplesPerPixel = 100;
    int maxDepth = 50;
    bool useBoundingVolumeHierarchy = true;
    bool usePythonRendering = true;
};

class Scene{
public:
    Scene();
    ~Scene();

    Camera* sceneCamera;
    std::vector<std::shared_ptr<IVirtualObject>> sceneObjects;
    std::vector<std::shared_ptr<IRenderable>> renderables;
    int selectedRenderable = -1;

    bool pythonRenderingUsed = true;
    std::shared_ptr<std::function<void()>> openGLRenderFunction;
    RenderSettings renderSettings;
    Image renderedImage = Image();

    void Initialize();

    void Render(std::function<void()> preProcessing = nullptr, std::function<Image()> renderStep = nullptr, std::function<void()> postProcessing = nullptr, std::function<void()> fallBack = nullptr);
    void UpdateScene();
    bool SaveScene(const std::string& filePath);
    bool LoadScene(const std::string& filePath);
    bool SaveRenderedImage(const std::string& filePath);

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
    
    py::object pyAlgorithimModule;
    py::object pyRaytracingModule;
    py::object pySamplerModule;
    py::object pySceneModule;
    py::object pyCameraModule;
    py::object pyLuminanceModule;
    py::object pyGeometryModule;

    void CleanUp();
};