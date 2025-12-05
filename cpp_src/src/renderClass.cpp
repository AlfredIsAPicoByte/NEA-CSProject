#include "renderClass.h"

Scene::Scene()
{
    AppendMessage("Scene created.");
}

Scene::~Scene()
{
    CleanUp();
    AppendMessage("Scene destroyed.");
}

void Scene::Initialize()
{
    PythonManager::Instance().AddModulePath("src");

    pyAlgorithimModule = PythonManager::Instance().LoadModule("baseAlgorithm");
    pyRaytracingModule = PythonManager::Instance().LoadModule("raytracing");
    pySamplerModule = PythonManager::Instance().LoadModule("sampler");
    pySceneModule = PythonManager::Instance().LoadModule("scene");
    pyCameraModule = PythonManager::Instance().LoadModule("camera");
    pyLuminanceModule = PythonManager::Instance().LoadModule("luminance");
    pyGeometryModule = PythonManager::Instance().LoadModule("geometry");
}


void Scene::Render(std::function<void()> processing, std::function<void()> renderStep, std::function<void()> postProcessing, std::function<void()> fallBack)
{
    if (pythonRenderingUsed) {
        // Call Python rendering functions here
        try {
            if (processing) processing();
            if (renderStep) renderStep();
            if (postProcessing) postProcessing();
        } catch (const std::exception& e) {
            AppendPythonError(std::string("Python rendering error: ") + e.what());
            if (fallBack) fallBack();
        }
    } else {
        // Use openGL rendering
        if (openGLRenderFunction) (*openGLRenderFunction)();
    }
}

void Scene::UpdateScene()
{
    // Update scene logic here
}

void Scene::LoadModel(const std::string& modelPath, Shader& shader)
{
    //
}

void Scene::SetCamera(Camera* camera)
{
    sceneCamera = camera;
}

void Scene::SetOpenGLRenderFunction(std::shared_ptr<std::function<void()>> renderFunc)
{
    openGLRenderFunction = renderFunc;
}

void Scene::CleanUp()
{
    renderables.clear();
}