#include "renderClass.h"

Scene::Scene()
    : shaderProgram(nullptr), camera(nullptr)
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
    
}

void Scene::update()
{

}

void Scene::cleanup()
{

}

void Scene::loadModel(const std::string& modelPath)
{

}

void Scene::loadTexture(const std::string& texturePath)
{

}

void Scene::setCamera(Camera* cam)
{
    sceneCamera = cam;
}