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


void Scene::Render(std::function<void()> preProcessing, std::function<Image()> renderStep, std::function<void()> postProcessing, std::function<void()> fallBack)
{
    if (pythonRenderingUsed) {
        // Call Python rendering functions here
        try {
            if (preProcessing) preProcessing();
            if (renderStep) renderedImage = renderStep();
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

bool Scene::SaveScene(const std::string& filePath)
{
    // Implement scene saving logic here
    return false;
}

bool Scene::LoadScene(const std::string& filePath)
{
    // Implement scene loading logic here
    return false;
}

bool Scene::SaveRenderedImage(const std::string& filePath)
{
    std::vector<uint8_t> pixels(renderSettings.imageWidth * renderSettings.imageHeight * 3);
    glReadPixels(0, 0, renderSettings.imageWidth , renderSettings.imageHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    // flip rows if needed
    stbi_write_png(filePath.c_str(), renderSettings.imageWidth , renderSettings.imageHeight, 3, pixels.data(), renderSettings.imageWidth  * 3);
    return true;
}

void Scene::AddRenderable(std::shared_ptr<IRenderable> renderable)
{
    renderables.push_back(renderable);
}

void Scene::CleanUp()
{
    renderables.clear();
}