#include "Camera.h"

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
	glm::mat4 view = glm::lookAt(Position, Position + Forward, Up);
	glm::mat4 projection = glm::mat4(1.0f);

	// Adds perspective to the scene
	if (type == PERSPECTIVE) {
		projection = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
	}
	else if (type == ORTHOGRAPHIC) {
		projection = glm::ortho(-aspectRatio * fov, aspectRatio * fov, -fov, fov, nearPlane, farPlane);
	}

	// Sets new camera matrix
	cameraMatrix = projection * view;
}

void Camera::Resize(int width, int height)
{
	windowWidth = width;
	windowHeight = height;
	aspectRatio = static_cast<float>(windowWidth) / static_cast<float>(windowHeight);
}

void Camera::Matrix(Shader& shader, const char* uniform)
{
	// Exports camera matrix to the shader
	glUniformMatrix4fv(glGetUniformLocation(shader.ID, uniform), 1, GL_FALSE, glm::value_ptr(cameraMatrix));
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
		default:
			// Unknown mode
			break;
	}
}

void Camera::FirstPersonMovement(GLFWwindow* window, double deltaTime)
{
    float _speed = speed * static_cast<float>(deltaTime);
	
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_SHIFT, false, true, [&]() {
		_speed *= speedFactor;
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_CONTROL, false, true, [&]() {
		_speed /= speedFactor;
	});

	// Handles key inputs
    InputManager::Instance(window).doWhenKey(GLFW_KEY_W, false, true, [&]() {
        Position += _speed * Forward;
	});
    InputManager::Instance(window).doWhenKey(GLFW_KEY_S, false, true, [&]() {
        Position -= _speed * Forward;
	});
    InputManager::Instance(window).doWhenKey(GLFW_KEY_A, false, true, [&]() {
        Position -= _speed * Right;
	});
    InputManager::Instance(window).doWhenKey(GLFW_KEY_D, false, true, [&]() {
        Position += _speed * Right;
	});
    InputManager::Instance(window).doWhenKey(GLFW_KEY_E, false, true, [&]() {
        Position += _speed * Up;
	});
    InputManager::Instance(window).doWhenKey(GLFW_KEY_Q, false, true, [&]() {
        Position -= _speed * Up;
	});

	// Handles mouse inputs
	InputManager::Instance(window).doWhenKey(GLFW_MOUSE_BUTTON_RIGHT, true, true, [&]() {
		// Hides mouse cursor
		if (!cursorHidden) {
			InputManager::Instance(window).setCursorVisibility(false);
			cursorHidden = true;
		}

		// Prevents camera from jumping on the first click
		if (firstClick)
		{
			InputManager::Instance(window).setMousePosition(windowWidth / 2, windowHeight / 2);
			firstClick = false;
		}

		// Stores the coordinates of the cursor
		double mouseX;
		double mouseY;
		// Fetches the coordinates of the cursor
		InputManager::Instance(window).getMousePosition(mouseX, mouseY);

		// Normalizes and shifts the coordinates of the cursor such that they begin in the middle of the screen
		// and then "transforms" them into degrees 
		float rotX = mouseSensitivity * (float)(mouseY - (windowWidth / 2)) / windowWidth;
		float rotY = mouseSensitivity * (float)(mouseX - (windowHeight / 2)) / windowHeight;

		// Calculates upcoming vertical change in the Orientation
		// Apply pitch rotation using a matrix rotate and test against the world up vector.
		glm::vec3 pitchedOrientation = glm::normalize(glm::mat3(glm::rotate(glm::mat4(1.0f), glm::radians(-rotX), Right)) * Forward);

		// Decides whether or not the next vertical orientation is legal or not
		if (abs(glm::angle(pitchedOrientation, Up) - glm::radians(90.0f)) <= glm::radians(85.0f))
		{
			Forward = pitchedOrientation;
		}

		// Rotates the orientation left and right (yaw) around Up
		Forward = glm::normalize(glm::mat3(glm::rotate(glm::mat4(1.0f), glm::radians(-rotY), Up)) * Forward);
		Right = glm::normalize(glm::cross(Forward, WorldUp));
		Up = glm::normalize(glm::cross(Right, Forward));

		// Sets mouse cursor to the middle of the screen so that it doesn't end up roaming around
		InputManager::Instance(window).setMousePosition(windowWidth / 2, windowHeight / 2);
	});
	
	InputManager::Instance(window).doWhenKey(GLFW_MOUSE_BUTTON_RIGHT, true, false, [&]() {
		// Unhides cursor since camera is not looking around anymore
		if (cursorHidden) {
			InputManager::Instance(window).setCursorVisibility(true);
			cursorHidden = false;
		}

		// Makes sure the next time the camera looks around it doesn't jump
		firstClick = true;
	});
}

void Camera::SetPlaneTarget(glm::vec3 target)
{
	planePitch = glm::degrees(asin(target.y - Position.y));
	planeYaw = glm::degrees(atan2(target.x - Position.x, -target.z - Position.z));
	planeRoll = 0.0f;
}

void Camera::PlaneMovement(GLFWwindow* window, double deltaTime)
{
	float rotationSpeed = planeRotateSpeed * static_cast<float>(deltaTime); // Rotation speed

	// Handles key inputs
	InputManager::Instance(window).doWhenKey(GLFW_KEY_W, false, true, [&]() {
		planePitch -= rotationSpeed * 10.0f * (invertY ? -1.0f : 1.0f);
		if (planePitch < -89.0f) planePitch = -89.0f;
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_S, false, true, [&]() {
		planePitch += rotationSpeed * 10.0f * (invertY ? -1.0f : 1.0f);
		if (planePitch > 89.0f) planePitch = 89.0f;
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_D, false, true, [&]() {
		if (planeYaw < -180.0f) planeYaw += 360.0f;
		planeYaw -= rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f);
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_A, false, true, [&]() {
		if (planeYaw > 180.0f) planeYaw -= 360.0f;
		planeYaw += rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f);
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_Q, false, true, [&]() {
		if (planeRoll < -180.0f) planeRoll += 360.0f;
		planeRoll -= rotationSpeed * 10.0f;
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_E, false, true, [&]() {
		if (planeRoll > 180.0f) planeRoll -= 360.0f;
		planeRoll += rotationSpeed * 10.0f;
	});
	
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_SHIFT, false, true, [&]() {
		if (!isPlanePowerIncreasing) {
			isPlanePowerIncreasing = true;
			return;
		}

		planePower += planePowerIncrement * static_cast<float>(deltaTime);
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_CONTROL, false, true, [&]() {
		if (isPlanePowerIncreasing) {
			isPlanePowerIncreasing = false;
			return;
		}
		
		planePower -= planePowerIncrement * static_cast<float>(deltaTime);
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_ALT, false, true, [&]() {
		if (!isPlaneActivelyStoping) {
			isPlaneActivelyStoping = true;
			return;
		}

		if (planePower > 0) {
			planePower -= planePowerIncrement * static_cast<float>(deltaTime);
		}
		else if (planePower < 0) {
			planePower += planePowerIncrement * static_cast<float>(deltaTime);
		}
		
		if (abs(planePower) < 0.01f) {
			planePower = 0.0f;
		}
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_ALT, false, false, [&]() {
		if (isPlaneActivelyStoping) {
			isPlaneActivelyStoping = false;
			return;
		}
	});
	
	if (planePower > planeMaxPower) {
		planePower = planeMaxPower;
	}
	else if (planePower < planeMinPower) {
		planePower = planeMinPower;
	}

	// Update Orientation based on pitch, yaw, and roll
	// Build a quaternion from Euler angles (pitch -> X, yaw -> Y, roll -> Z)
	// NOTE: ensure the order matches (pitch, yaw, roll)
	glm::quat q = glm::quat(glm::vec3(glm::radians(planePitch), glm::radians(planeYaw), glm::radians(planeRoll)));

	// Rotate the forward vector by the quaternion to get the facing direction
	Forward = q * glm::vec3(0.0f, 0.0f, -1.0f);
	
	// Recalculate Right vector
	Right = q * glm::vec3(1.0f, 0.0f, 0.0f);

	// Apply roll to Up as well so lookAt will reflect roll tilt
	Up = q * WorldUp; // Use the stored world up vector as the base
	
	// Move the camera forward in the direction it's facing
	Position += planePower * static_cast<float>(deltaTime) * Forward;
}

void Camera::SelectOrbitTarget(const glm::vec3& target)
{
	orbitTarget = target;
}

void Camera::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    if (!cam) return;

    if (yoffset > 0) {
        cam->orbitDistance -= cam->orbitZoomSpeed;
    } else {
        cam->orbitDistance += cam->orbitZoomSpeed;
    }

    if (cam->orbitDistance < cam->orbitMinDistance) cam->orbitDistance = cam->orbitMinDistance;
    if (cam->orbitDistance > cam->orbitMaxDistance) cam->orbitDistance = cam->orbitMaxDistance;
}

void Camera::OrbitMovement(GLFWwindow* window, double deltaTime)
{
	// ensure the window user pointer points to this Camera instance
	glfwSetWindowUserPointer(window, this);
	// set the scroll callback to the static Camera method
	glfwSetScrollCallback(window, Camera::scroll_callback);
	
	InputManager::Instance(window).doWhenKey(GLFW_MOUSE_BUTTON_RIGHT, true, true, [&]() {
		if (!cursorHidden) {
			InputManager::Instance(window).setCursorVisibility(false);
			cursorHidden = true;
		}

		if (firstClick) {
			InputManager::Instance(window).setMousePosition(windowWidth / 2, windowHeight / 2);
			firstClick = false;
		}

		double mouseX, mouseY;
		InputManager::Instance(window).getMousePosition(mouseX, mouseY);

		float mouse_dx = (float)(mouseX - (windowWidth / 2));
		float mouse_dy = (float)(mouseY - (windowHeight / 2));

		float orbitSpeed = mouseSensitivity * static_cast<float>(deltaTime);
		
		float yaw = orbitSpeed * mouse_dx / windowWidth;
		float pitch = orbitSpeed * mouse_dy / windowHeight;
		float max_pitch = glm::radians(89.0f);

		// Apply invertY to the tentative new pitch for clamping check
		float new_pitch = orbitPitch + pitch * (invertY ? -1.0f : 1.0f);
		if (abs(new_pitch) > max_pitch) {
			pitch = (new_pitch > 0 ? max_pitch : -max_pitch) - orbitPitch;
			pitch *= (invertY ? 1.0f : 1.0f) ; // keep sign handling consistent below
		}

		orbitPitch += pitch * (invertY ? -1.0f : 1.0f);
		orbitYaw += yaw * (invertX ? -1.0f : 1.0f);

		// Keep yaw within -pi..pi for numerical stability
		if (orbitYaw > glm::pi<float>()) orbitYaw -= glm::two_pi<float>();
		if (orbitYaw <= -glm::pi<float>()) orbitYaw += glm::two_pi<float>();

		// Clamp radius
		if (orbitDistance < orbitMinDistance) orbitDistance = orbitMinDistance;
		if (orbitDistance > orbitMaxDistance) orbitDistance = orbitMaxDistance;

		// Build rotation from orbitPitch and orbitYaw (roll handled separately so it is local to camera orientation)
		glm::quat q = glm::quat(glm::vec3(orbitPitch, orbitYaw, 0.0f));

		// Rotate a vector at distance along +Z by the quaternion to get camera offset (without roll)
		glm::vec3 offset = q * glm::vec3(0.0f, 0.0f, orbitDistance);

		// Apply roll as a rotation around the camera-target axis (i.e., local to the camera orientation).
		// This makes roll affect the overall motion of the orbit rather than being a global/world Z rotation.
		InputManager::Instance(window).doWhenKey(GLFW_KEY_Q, false, true, [&]() {
			orbitRoll -= orbitSpeed;
		});
		InputManager::Instance(window).doWhenKey(GLFW_KEY_E, false, true, [&]() {
			orbitRoll += orbitSpeed;
		});

		if (fabs(orbitRoll) > 1e-6f) {
			glm::quat qroll = glm::angleAxis(orbitRoll, Forward) * static_cast<float>(deltaTime);
			offset = qroll * offset;
		}

		float x = orbitTarget.x + offset.x;
		float y = orbitTarget.y + offset.y;
		float z = orbitTarget.z + offset.z;

		Position = glm::vec3(x, y, z);

		LookAt(orbitTarget);
		
		InputManager::Instance(window).setMousePosition(windowWidth / 2, windowHeight / 2);
	});
	
	InputManager::Instance(window).doWhenKey(GLFW_MOUSE_BUTTON_RIGHT, true, false, [&]() {
		// Unhides cursor since camera is not looking around anymore
		if (cursorHidden) {
			InputManager::Instance(window).setCursorVisibility(true);
			cursorHidden = false;
		}

		// Makes sure the next time the camera looks around it doesn't jump
		firstClick = true;
	});
}

void Camera::LookAt(const glm::vec3& target)
{
	Forward = glm::normalize(target - Position);
}