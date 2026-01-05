#include "cameraClass.h"

Camera::Camera(int width, int height, glm::vec3 position, glm::vec3 forward, glm::vec3 worldUp)
{
    Resize(width, height);
    Position = position;
    Forward = glm::normalize(forward);
    WorldUp = worldUp;

    Right = glm::normalize(glm::cross(Forward, WorldUp));
    Up = glm::normalize(glm::cross(Right, Forward));
}

void Camera::updateMatrix()
{
    // FIX: Store these in the class members, not just local variables
    viewMatrix = glm::lookAt(Position, Position + Forward, Up);
    projectionMatrix = glm::mat4(1.0f);

    if (type == PERSPECTIVE) {
        projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
    }
    else if (type == ORTHOGRAPHIC) {
        projectionMatrix = glm::ortho(-aspectRatio * fov, aspectRatio * fov, -fov, fov, nearPlane, farPlane);
    }

    cameraMatrix = projectionMatrix * viewMatrix;
}

void Camera::Resize(int width, int height)
{
    if (height == 0) height = 1; // Safety check
    windowWidth = width;
    windowHeight = height;
    aspectRatio = static_cast<float>(windowWidth) / static_cast<float>(windowHeight);
}

void Camera::SetModelMatrixUniform(Shader& shader, const char* uniform)
{
    shader.setMat4(uniform, cameraMatrix);
}

void Camera::Move(GLFWwindow* window, double deltaTime)
{
    switch (mode) {
        case FIRST_PERSON:
            FirstPersonMovement(window, deltaTime);
            break;
        case PLANE:
            PlaneMovement(window, deltaTime);
            break;
        case ORBIT:
            OrbitMovement(window, deltaTime);
            break;
    }
}

void Camera::FirstPersonMovement(GLFWwindow* window, double deltaTime)
{
    float _speed = speed * static_cast<float>(deltaTime);
    
    // Check Inputs
    InputManager& input = InputManager::Instance(window);

    input.doWhenKey(GLFW_KEY_LEFT_SHIFT, true, [&]() { _speed *= speedFactor; });
    input.doWhenKey(GLFW_KEY_LEFT_CONTROL, true, [&]() { _speed /= speedFactor; });

    input.doWhenKey(GLFW_KEY_W, true, [&]() { Position += _speed * Forward; });
    input.doWhenKey(GLFW_KEY_S, true, [&]() { Position -= _speed * Forward; });
    input.doWhenKey(GLFW_KEY_A, true, [&]() { Position -= _speed * Right; });
    input.doWhenKey(GLFW_KEY_D, true, [&]() { Position += _speed * Right; });
    input.doWhenKey(GLFW_KEY_E, true, [&]() { Position += _speed * Up; });
    input.doWhenKey(GLFW_KEY_Q, true, [&]() { Position -= _speed * Up; });

    // Mouse Look
    input.doWhenMouseKey(GLFW_MOUSE_BUTTON_RIGHT, true, [&]() {
        input.setCursorVisibility(false);

        if (firstClick) {
            input.setMousePosition(windowWidth / 2, windowHeight / 2);
            firstClick = false;
        }

        double mouseX, mouseY;
        input.getMousePosition(mouseX, mouseY);

        float rotX = mouseSensitivity * (float)(mouseY - (windowHeight / 2)) / windowHeight;
        float rotY = mouseSensitivity * (float)(mouseX - (windowWidth / 2)) / windowWidth;

        if (invertY) rotX = -rotX;
        if (invertX) rotY = -rotY;

        // Pitch (Vertical)
        glm::vec3 pitchedOrientation = glm::normalize(glm::rotate(Forward, glm::radians(-rotX), Right));
        
        // Clamp pitch to avoid gimbal lock (flipping over)
        if (abs(glm::angle(pitchedOrientation, WorldUp) - glm::radians(90.0f)) <= glm::radians(85.0f)) {
            Forward = pitchedOrientation;
        }

        // Yaw (Horizontal) - Rotate around WorldUp
        Forward = glm::normalize(glm::rotate(Forward, glm::radians(-rotY), WorldUp));
        
        // Re-orthogonalize vectors
        Right = glm::normalize(glm::cross(Forward, WorldUp));
        Up = glm::normalize(glm::cross(Right, Forward));

        input.setMousePosition(windowWidth / 2, windowHeight / 2);
    });
    
    input.doWhenMouseKey(GLFW_MOUSE_BUTTON_RIGHT, false, [&]() {
        input.setCursorVisibility(true);
        firstClick = true;
    });
}

void Camera::SetPlaneTarget(glm::vec3 target)
{
    glm::vec3 direction = glm::normalize(target - Position);
    planePitch = glm::degrees(asin(direction.y));
    planeYaw = glm::degrees(atan2(direction.z, direction.x)); // Note: atan2(z, x) corresponds to yaw in many systems
    planeRoll = 0.0f;
}

void Camera::PlaneMovement(GLFWwindow* window, double deltaTime)
{
    float dt = static_cast<float>(deltaTime);
    float rotationSpeed = planeRotateSpeed * dt;

    InputManager& input = InputManager::Instance(window);

    input.doWhenKey(GLFW_KEY_W, true, [&]() { planePitch -= rotationSpeed * 10.0f * (invertY ? -1.0f : 1.0f); });
    input.doWhenKey(GLFW_KEY_S, true, [&]() { planePitch += rotationSpeed * 10.0f * (invertY ? -1.0f : 1.0f); });
    input.doWhenKey(GLFW_KEY_D, true, [&]() { planeYaw -= rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f); });
    input.doWhenKey(GLFW_KEY_A, true, [&]() { planeYaw += rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f); });
    input.doWhenKey(GLFW_KEY_Q, true, [&]() { planeRoll -= rotationSpeed * 10.0f; });
    input.doWhenKey(GLFW_KEY_E, true, [&]() { planeRoll += rotationSpeed * 10.0f; });

    // Speed Control
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
         planePower += planePowerIncrement * dt;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
         planePower -= planePowerIncrement * dt;
    }
    
    // Active braking
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) {
        if (planePower > 0) planePower -= planePowerIncrement * dt * 2.0f;
        else if (planePower < 0) planePower += planePowerIncrement * dt * 2.0f;
        if (abs(planePower) < 0.1f) planePower = 0.0f;
    }

    planePower = glm::clamp(planePower, planeMinPower, planeMaxPower);
    planePitch = glm::clamp(planePitch, -89.0f, 89.0f);

    // Quaternion Rotation: Yaw (Global Y) * Pitch (Local X) * Roll (Local Z)
    glm::quat qPitch = glm::angleAxis(glm::radians(planePitch), glm::vec3(1, 0, 0));
    glm::quat qYaw = glm::angleAxis(glm::radians(planeYaw), glm::vec3(0, 1, 0)); 
    glm::quat qRoll = glm::angleAxis(glm::radians(planeRoll), glm::vec3(0, 0, 1));

    // Order depends on desired mechanics. Usually Yaw * Pitch * Roll.
    glm::quat orientation = qYaw * qPitch * qRoll;

    Forward = glm::rotate(orientation, glm::vec3(0.0f, 0.0f, -1.0f));
    Up = glm::rotate(orientation, glm::vec3(0.0f, 1.0f, 0.0f));
    Right = glm::cross(Forward, Up);

    Position += Forward * planePower * dt;
}

void Camera::SelectOrbitTarget(const glm::vec3& target)
{
    orbitTarget = target;
    // Calculate initial pitch/yaw based on current position relative to target
    glm::vec3 dir = glm::normalize(Position - target);
    orbitDistance = glm::distance(Position, target);
    orbitPitch = asin(dir.y);
    orbitYaw = atan2(dir.x, dir.z);
}

void Camera::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // Safety: ensure the pointer is actually a Camera
    Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    if (cam) {
        cam->orbitDistance -= static_cast<float>(yoffset) * cam->orbitZoomSpeed;
        cam->orbitDistance = glm::clamp(cam->orbitDistance, cam->orbitMinDistance, cam->orbitMaxDistance);
    }
}

void Camera::OrbitMovement(GLFWwindow* window, double deltaTime)
{
    // WARNING: This overwrites UserPointer. If InputManager needs this, this will break InputManager.
    // Ideally, InputManager should handle scroll events and pass them here.
    if (glfwGetWindowUserPointer(window) != this) {
        glfwSetWindowUserPointer(window, this);
        glfwSetScrollCallback(window, Camera::scroll_callback);
    }

    InputManager& input = InputManager::Instance(window);
    
    // Handle Roll Keys (Independent of mouse click for better UX, or inside if preferred)
    float dt = static_cast<float>(deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) orbitRoll -= orbitSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) orbitRoll += orbitSpeed * dt;

    input.doWhenMouseKey(GLFW_MOUSE_BUTTON_RIGHT, true, [&]() {
        input.setCursorVisibility(false);

        if (firstClick) {
            input.setMousePosition(windowWidth / 2, windowHeight / 2);
            firstClick = false;
        }

        double mouseX, mouseY;
        input.getMousePosition(mouseX, mouseY);

        float dx = (float)(mouseX - (windowWidth / 2));
        float dy = (float)(mouseY - (windowHeight / 2));

        orbitYaw += dx * mouseSensitivity * 0.005f * dt * (invertX ? -1.0f : 1.0f);
        orbitPitch += dy * mouseSensitivity * 0.005f * dt * (invertY ? -1.0f : 1.0f);

        // Clamp Pitch
        orbitPitch = glm::clamp(orbitPitch, -1.5f, 1.5f); // Approx -85 to 85 degrees

        input.setMousePosition(windowWidth / 2, windowHeight / 2);
    });

    input.doWhenMouseKey(GLFW_MOUSE_BUTTON_RIGHT, false, [&]() {
        input.setCursorVisibility(true);
        firstClick = true;
    });

    // Calculate Position based on Spherical Coordinates
    // Quaternion for rotation: Yaw -> Pitch -> Roll
    glm::quat qYaw = glm::angleAxis(orbitYaw, glm::vec3(0, 1, 0));
    glm::quat qPitch = glm::angleAxis(orbitPitch, glm::vec3(1, 0, 0));
    glm::quat qRoll = glm::angleAxis(orbitRoll, glm::vec3(0, 0, 1));

    glm::quat orientation = qYaw * qPitch * qRoll;

    // Calculate offset from target (assuming camera looks down -Z)
    glm::vec3 offset = orientation * glm::vec3(0.0f, 0.0f, orbitDistance);

    Position = orbitTarget + offset;

    // Manually set vectors based on quaternion (Standard LookAt overrides Roll)
    // Forward is direction FROM camera TO target
    Forward = glm::normalize(orbitTarget - Position);
    
    // Up is determined by the orientation quaternion
    Up = glm::normalize(orientation * glm::vec3(0, 1, 0));
    Right = glm::normalize(glm::cross(Forward, Up));
}

void Camera::LookAt(const glm::vec3& target)
{
    Forward = glm::normalize(target - Position);
    Right = glm::normalize(glm::cross(Forward, WorldUp));
    Up = glm::normalize(glm::cross(Right, Forward));
}

void Camera::ResetOrbit()
{
    orbitPitch = 0.0f;
    orbitYaw = 0.0f;
    orbitRoll = 0.0f;
    orbitDistance = 10.0f;
}

void Camera::ResetPlane()
{
    planePitch = 0.0f;
    planeYaw = -90.0f;
    planeRoll = 0.0f;
    planePower = 0.0f;
}

void Camera::Reset(glm::vec3 position, glm::vec3 forward, glm::vec3 worldUp)
{
    Position = position;
    Forward = glm::normalize(forward);
    WorldUp = worldUp;
    Right = glm::normalize(glm::cross(Forward, WorldUp));
    Up = glm::normalize(glm::cross(Right, Forward));
}

void Camera::SerializeFields(json& j) const
{
    j["Position"] = { Position.x, Position.y, Position.z };
    j["WorldUp"] = { WorldUp.x, WorldUp.y, WorldUp.z };
    j["Forward"] = { Forward.x, Forward.y, Forward.z };
    j["type"] = static_cast<int>(type);
    j["mode"] = static_cast<int>(mode);
    j["fov"] = fov;
    j["speed"] = speed;
    j["sensitivity"] = mouseSensitivity;
    
    // Save Orbit State
    j["orbit"] = {
        {"pitch", orbitPitch}, {"yaw", orbitYaw}, {"roll", orbitRoll},
        {"distance", orbitDistance}, {"target", {orbitTarget.x, orbitTarget.y, orbitTarget.z}}
    };

    // Save Plane State
    j["plane"] = {
        {"pitch", planePitch}, {"yaw", planeYaw}, {"roll", planeRoll}, {"power", planePower}
    };
}

void Camera::DeserializeFields(const json& j)
{
    Camera t = *this; // Temporary copy in case of partial failure
    if(j.contains("Position")) t.Position = glm::vec3(j["Position"][0], j["Position"][1], j["Position"][2]);
    if(j.contains("WorldUp")) t.WorldUp = glm::vec3(j["WorldUp"][0], j["WorldUp"][1], j["WorldUp"][2]);
    if(j.contains("Forward")) t.Forward = glm::vec3(j["Forward"][0], j["Forward"][1], j["Forward"][2]);
    
    // Recalculate dependent vectors
    t.Right = glm::normalize(glm::cross(t.Forward, t.WorldUp));
    t.Up = glm::normalize(glm::cross(t.Right, t.Forward));

    if(j.contains("type")) t.type = static_cast<CameraType>(j["type"]);
    if(j.contains("fov")) t.fov = j["fov"];
    if(j.contains("speed")) t.speed = j["speed"];
    if(j.contains("sensitivity")) t.mouseSensitivity = j["sensitivity"];
    if(j.contains("orbit")) {
        t.orbitPitch = j["orbit"]["pitch"];
        t.orbitYaw = j["orbit"]["yaw"];
        t.orbitRoll = j["orbit"]["roll"];
        t.orbitDistance = j["orbit"]["distance"];
        t.orbitTarget = glm::vec3(j["orbit"]["target"][0], j["orbit"]["target"][1], j["orbit"]["target"][2]);
    }

    if(j.contains("plane")) {
        t.planePitch = j["plane"]["pitch"];
        t.planeYaw = j["plane"]["yaw"];
        t.planeRoll = j["plane"]["roll"];
        t.planePower = j["plane"]["power"];
    }
}