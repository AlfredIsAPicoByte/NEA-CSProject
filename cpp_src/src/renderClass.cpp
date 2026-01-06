#include "renderClass.h"

Scene::Scene()
{
    renderSettings = std::make_shared<RenderSettings>();
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

    pyAlgorithmModule = PythonManager::Instance().LoadModule("baseAlgorithm");
    pyRaytracingModule = PythonManager::Instance().LoadModule("raytracing");
    pySamplerModule = PythonManager::Instance().LoadModule("sampler");
    pySceneModule = PythonManager::Instance().LoadModule("scene");
    pyCameraModule = PythonManager::Instance().LoadModule("camera");
    pyLuminanceModule = PythonManager::Instance().LoadModule("luminance");
    pyGeometryModule = PythonManager::Instance().LoadModule("geometry");
}


void Scene::Render(std::function<void()> preProcessing, std::function<Image()> renderStep, std::function<void()> postProcessing, std::function<void()> fallBack)
{
    if (preProcessing) preProcessing();

    if (renderStep) {
        // run provided render step (e.g. Python renderer)
        Image img = renderStep();
        (void)img; // ignore if caller doesn't need stored image here
    } else if (openGLRenderFunction && !renderSettings->usePythonRendering) {
        (*openGLRenderFunction)();
    } else if (fallBack) {
        fallBack();
    }

    if (postProcessing) postProcessing();
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
    json j;
    SerializeFields(j);

    // Implement scene saving logic here
    std::ofstream file(filePath);
    if (file.is_open()) {
        file << j.dump(4);
        file.close();
        return true;
    }

    AppendError("Failed to open file for saving: " + filePath);
    return false;
}

void Scene::SerializeFields(json& j) const
{
    j["name"] = name;
    j["active_camera"] = sceneCamera->ToJSON() ? sceneCamera->GetLocalID() : -1;
    j["selected_renderable"] = selectedRenderable;
    renderSettings->SerializeFields(j["render_settings"]);

    j["renderables"] = json::array();
    for (const auto& r : renderables) {
        if (!r) { j["renderables"].push_back(nullptr); continue; }
        // try to treat renderable as IVirtualObject to get name
        auto asVO = std::dynamic_pointer_cast<IVirtualObject>(r);
        if (asVO) {
            j["renderables"].push_back(asVO->GetLocalID());
        }
        else j["renderables"].push_back("renderable");
    }
    j["objects"] = json::array();
    for (const auto& obj : sceneObjects) {
        if (!obj) { j["objects"].push_back(nullptr); continue; }
        // IVirtualObject is expected to have a public name field
        j["objects"].push_back(obj->GetLocalID());
    }
    j["constructor"] = "Scene";
}

bool Scene::LoadScene(const std::string& filePath)
{
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) return false;

    json j;
    try {
        ifs >> j;
    } catch (...) {
        AppendError("Failed to parse scene JSON from file: " + filePath);
        return false;
    }
    ifs.close();

    DeserializeFields(j);
    return true;
}

void Scene::DeserializeFields(const json& j)
{
    if (j.contains("name")) {
        name = j["name"];
    }
    if (j.contains("active_camera")) {
        int camID = j["active_camera"];
        // Find camera by ID in sceneObjects
        for (const auto& obj : sceneObjects) {
            if (!obj) continue;
            if (obj->GetLocalID() == camID) {
                sceneCamera = dynamic_cast<Camera*>(obj.get());
                break;
            }
        }
    }
    if (j.contains("selected_renderable")) {
        selectedRenderable = j["selected_renderable"];
    }
    if (j.contains("render_settings")) {
        const auto& rs = j["render_settings"];
        if (rs.contains("image_width")) renderSettings->imageWidth = rs["image_width"];
        if (rs.contains("image_height")) renderSettings->imageHeight = rs["image_height"];
        if (rs.contains("use_python_rendering")) renderSettings->usePythonRendering = rs["use_python_rendering"];
    }

    if (j.contains("renderables")) {
        renderables.clear();
        for (const auto& rData : j["renderables"]) {
            if (rData.is_null()) {
                renderables.push_back(nullptr);
                continue;
            }
            // Find renderable by name and constructor
            std::shared_ptr<IRenderable> foundRenderable = nullptr;
            std::string rName = rData.is_string() ? rData.get<std::string>() : "";
            std::string constructor = rData.contains("constructor") ? rData["constructor"].get<std::string>() : "";

            if (strcmp(constructor.c_str(), "ModelAdapter") == 0) {
                foundRenderable = std::make_shared<ModelAdapter>(nullptr, 0);
            }
            else if (strcmp(constructor.c_str(), "Mesh") == 0) {
                foundRenderable = std::make_shared<Mesh>();
            }
            else {
                AppendWarning("Unknown renderable constructor: " + constructor);
                continue;
            }

            foundRenderable->FromJSON(rData);
            AddRenderable(foundRenderable);
        }
    }

    if (j.contains("objects")) {
        sceneObjects.clear();
        for (const auto& oData : j["objects"]) {
            if (oData.is_null()) {
                sceneObjects.push_back(nullptr);
                continue;
            }
            // Find object by name and constructor
            std::shared_ptr<IVirtualObject> foundObject = nullptr;
            std::string oName = oData.is_string() ? oData.get<std::string>() : "";
            std::string constructor = oData.contains("constructor") ? oData["constructor"].get<std::string>() : "";

            if (strcmp(constructor.c_str(), "Camera") == 0) {
                auto camera = std::make_shared<Camera>(
                    800,   // or specific width
                    400,  // or specific height
                    glm::vec3(0.0f, 0.0f, 3.0f),  // position
                    glm::vec3(0.0f, 0.0f, -1.0f)  // forward direction
                );
                foundObject = camera;
            }
            else {
                AppendWarning("Unknown object constructor: " + constructor);
                continue;
            }

            foundObject->FromJSON(oData);
            AddObject(foundObject);
        }
    }
}

bool Scene::SaveRenderedImage(const std::string& filePath)
{
    std::vector<uint8_t> pixels(renderSettings->imageWidth * renderSettings->imageHeight * 3);
    glReadPixels(0, 0, renderSettings->imageWidth , renderSettings->imageHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    // flip rows if needed
    stbi_write_png(filePath.c_str(), renderSettings->imageWidth , renderSettings->imageHeight, 3, pixels.data(), renderSettings->imageWidth * 3);
    return true;
}

void Scene::AddRenderable(std::shared_ptr<IRenderable> renderable)
{
    renderables.push_back(renderable);
}

std::shared_ptr<IRenderable> Scene::GetRenderable(const std::string& name)
{
    for (auto &r : renderables) {
        if (r && r->GetName() == name) return r;
    }
    return nullptr;
}

std::shared_ptr<IRenderable> Scene::GetRenderable(int index)
{
    if (index >= 0 && index < static_cast<int>(renderables.size()))
        return renderables[index];
    return nullptr;
}

void Scene::RemoveRenderable(const std::string& name)
{
    for (auto it = renderables.begin(); it != renderables.end(); ++it) {
        if (*it && (*it)->GetName() == name) {
            renderables.erase(it);
            selectedRenderable = -1;
            return;
        }
    }
}

void Scene::RemoveRenderable(int index)
{
    if (index >= 0 && index < static_cast<int>(renderables.size())) {
        renderables.erase(renderables.begin() + index);
        selectedRenderable = -1;
    }
}

void Scene::ClearRenderables()
{
    renderables.clear();
    selectedRenderable = -1;
}

void Scene::AddObject(std::shared_ptr<IVirtualObject> object)
{
    if (object) sceneObjects.push_back(object);
}

std::shared_ptr<IVirtualObject> Scene::GetObject(const std::string& name)
{
    for (auto &obj : sceneObjects) {
        if (!obj) continue;
        try {
            auto j = obj->ToJSON();
            if (j.contains("name") && j["name"].is_string() && j["name"].get<std::string>() == name)
                return obj;
        } catch (...) {
            // ignore objects that can't serialize
        }
    }
    return nullptr;
}

std::shared_ptr<IVirtualObject> Scene::GetObject(int index)
{
    if (index >= 0 && index < static_cast<int>(sceneObjects.size()))
        return sceneObjects[index];
    return nullptr;
}

void Scene::RemoveObject(const std::string& name)
{
    for (auto it = sceneObjects.begin(); it != sceneObjects.end(); ++it) {
        if (!*it) continue;
        try {
            auto j = (*it)->ToJSON();
            if (j.contains("name") && j["name"].is_string() && j["name"].get<std::string>() == name) {
                sceneObjects.erase(it);
                return;
            }
        } catch (...) {
            // ignore and continue
        }
    }
}

void Scene::RemoveObject(int index)
{
    if (index >= 0 && index < static_cast<int>(sceneObjects.size()))
        sceneObjects.erase(sceneObjects.begin() + index);
}

void Scene::ClearObjects()
{
    sceneObjects.clear();
}

void Scene::CleanUp()
{
    renderables.clear();
}