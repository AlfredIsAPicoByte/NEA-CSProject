#include "renderClass.h"

Scene::Scene()
{
    AppendMessage("Scene created.");
}

Scene::~Scene()
{
    cleanup();
    AppendMessage("Scene destroyed.");
}

void Scene::initialize()
{
    PythonManager::Instance().AddModulePath("src");
    pyCamera = PythonManager::Instance().LoadModule("Camera");
    pySampler = PythonManager::Instance().LoadModule("Sampler");
    pyScene = PythonManager::Instance().LoadModule("Scene");
    PythonManager::Instance().AddModulePath("src/Algorithms");
    pyBaseAlgorithm = PythonManager::Instance().LoadModule("Base");
    pyRayTracer = PythonManager::Instance().LoadModule("Raytracer");
}

void Scene::render()
{
    if (pythonRenderingUsed) {
        pyScene.attr("render")(pyCamera, pySampler);
    } else {
        shaderProgram->Activate();
        for (const auto& renderable : renderables) {
            renderable->Draw(*shaderProgram, *sceneCamera);
        }
    }
}

void Scene::update()
{

}

void Scene::cleanup()
{
    renderables.clear();
}

void Scene::loadModel(const std::string& modelPath)
{
    
}

void Scene::setCamera(Camera* cam)
{
    sceneCamera = cam;
}