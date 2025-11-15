#include "Camera.h"

Camera::Camera(int width, int height, glm::vec3 position)
{
	Resize(width, height);
	Position = position;
	WorldUp = glm::vec3(0.0f, 1.0f, 0.0f);
	Up = WorldUp;
}

void Camera::updateMatrix()
{
	// Initializes matrices since otherwise they will be the null matrix
	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);

	// Makes camera look in the right direction from the right position
	view = glm::lookAt(Position, Position + Orientation, Up);
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

void Camera::updateOrientation()
{
	Forward = glm::normalize(Forward);
	Right = glm::normalize(glm::cross(Forward, WorldUp));
	Up = glm::normalize(glm::cross(Right, Forward));
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
    float _speed = speed * deltaTime; // Movement speed

	// Handles key inputs
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		Position += _speed * Orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		Position += _speed * -glm::normalize(glm::cross(Orientation, Up));
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		Position += _speed * -Orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		Position += _speed * glm::normalize(glm::cross(Orientation, Up));
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		Position += _speed * Up;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		Position += _speed * -Up;
	}
	
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
	{
		_speed = speed * speedFactor * deltaTime;
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
	{
		_speed = speed * deltaTime;
	}

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
		glm::vec3 newOrientation = glm::rotate(Orientation, glm::radians(-rotX), glm::normalize(glm::cross(Orientation, Up)));

		// Decides whether or not the next vertical Orientation is legal or not
		if (abs(glm::angle(newOrientation, Up) - glm::radians(90.0f)) <= glm::radians(85.0f))
		{
			Orientation = newOrientation;
		}

		// Rotates the orientation left and right (yaw) around Up
		Forward = glm::mat3(glm::rotate(glm::mat4(1.0f), glm::radians(-rotY), Up)) * Forward;
		updateOrientation();

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
	LookAt(target);

	planePitch = glm::degrees(asin(Forward.y));
	planeYaw = glm::degrees(atan2(Forward.x, -Forward.z));
	planeRoll = 0.0f;
}

void Camera::PlaneMovement(GLFWwindow* window, double deltaTime)
{
	float rotationSpeed = planeRotateSpeed * deltaTime; // Rotation speed

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
		planeYaw += rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f);
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_A, false, true, [&]() {
		if (planeYaw > 180.0f) planeYaw -= 360.0f;
		planeYaw -= rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f);
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_Q, false, true, [&]() {
		planeRoll -= rotationSpeed * 10.0f;
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_E, false, true, [&]() {
		planeRoll += rotationSpeed * 10.0f;
	});
	
	bool isIncreasing = true;
	bool isStoping = false;
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_SHIFT, false, true, [&]() {
		if (!isIncreasing) {
			isIncreasing = true;
			return;
		}

		planePower += planePowerIncrement * deltaTime;
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_CONTROL, false, true, [&]() {
		if (isIncreasing) {
			isIncreasing = false;
			return;
		}
		
		planePower -= planePowerIncrement * deltaTime;
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_ALT, false, true, [&]() {
		if (!isStoping) {
			isStoping = true;
			return;
		}

		if (planePower > 0) {
			planePower -= planePowerIncrement * deltaTime;
		}
		else if (planePower < 0) {
			planePower += planePowerIncrement * deltaTime;
		}
		
		if (abs(planePower) < 0.01f) {
			planePower = 0.0f;
		}
	});
	InputManager::Instance(window).doWhenKey(GLFW_KEY_LEFT_ALT, false, false, [&]() {
		if (isStoping) {
			isStoping = false;
			return;
		}
	});
	
	if (planePower > planeMaxPower) {
		planePower = planeMaxPower;
	}
	else if (planePower < planeMinPower) {
		planePower = planeMinPower;
	}

	// Updates Forward vector based on Euler Angles
	Forward.x = cos(glm::radians(planePitch)) * sin(glm::radians(planeYaw));
	Forward.y = sin(glm::radians(planePitch));
	Forward.z = -cos(glm::radians(planePitch)) * cos(glm::radians(planeYaw));
	Forward = glm::normalize(Forward);

	// Base Right/Up from world up
	Right = glm::normalize(glm::cross(Forward, WorldUp));
	Up = glm::normalize(glm::cross(Right, Forward));

	// Normalize roll to [-180,180] to avoid drifting angles
	if (planeRoll > 180.0f) planeRoll -= 360.0f;
	if (planeRoll <= -180.0f) planeRoll += 360.0f;

	// Applies roll rotation around Forward axis (planeRoll is in degrees)
	if (fabs(planeRoll) > 1e-6f) {
		float rollRad = glm::radians(planeRoll);
		glm::quat qroll = glm::angleAxis(rollRad, Forward);
		Right = glm::normalize(qroll * Right);
		Up = glm::normalize(qroll * Up);
	}

	// Do not call updateOrientation() here because it would recompute Up/Right and overwrite roll
	
	// Move the camera forward in the direction it's facing
	Position += planePower * static_cast<float>(deltaTime) * Orientation;
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

		float orbit_speed = mouseSensitivity * deltaTime;
		
		float yaw = orbit_speed * mouse_dx / windowWidth;

		float pitch = orbit_speed * mouse_dy / windowHeight;
		float max_pitch = glm::radians(89.0f);
		float new_pitch = orbitPitch + pitch;

		if (abs(new_pitch) > max_pitch) {
			pitch = (new_pitch > 0 ? max_pitch : -max_pitch) - orbitPitch;
		}

		orbitPitch += pitch * (invertY ? -1.0f : 1.0f);
		orbitYaw += yaw * (invertX ? -1.0f : 1.0f);
		
		// Clamp radius
		if (orbitDistance < orbitMinDistance) orbitDistance = orbitMinDistance;
		if (orbitDistance > orbitMaxDistance) orbitDistance = orbitMaxDistance;

		// Build rotation from orbitPitch and orbitYaw (no roll here)
		glm::quat q = glm::quat(glm::vec3(orbitPitch, orbitYaw, 0.0f));

		// Rotate a vector at distance along +Z by the quaternion to get camera offset (without roll)
		glm::vec3 offset = q * glm::vec3(0.0f, 0.0f, orbitDistance);

		// Update position and forward
		Position = orbitTarget + offset;
		LookAt(orbitTarget);

		// Recompute base Right/Up from the updated Forward
		Right = glm::normalize(glm::cross(Forward, WorldUp));
		Up = glm::normalize(glm::cross(Right, Forward));

		// Adjust roll via keys (registered each frame like other modes)
		InputManager::Instance(window).doWhenKey(GLFW_KEY_Q, false, true, [&]() {
			orbitRoll -= orbitSpeed * orbitRollSpeed;
		});
		InputManager::Instance(window).doWhenKey(GLFW_KEY_E, false, true, [&]() {
			orbitRoll += orbitSpeed * orbitRollSpeed;
		});

		// Normalize roll to [-180,180] to avoid drifting angles
		if (orbitRoll > 180.0f) orbitRoll -= 360.0f;
		if (orbitRoll <= -180.0f) orbitRoll += 360.0f;

		// Apply roll around the current Forward axis (now correct)
		if (fabs(orbitRoll) > 1e-6f) {
			float rollRad = glm::radians(orbitRoll);
			glm::quat qroll = glm::angleAxis(rollRad, Forward);
			Right = glm::normalize(qroll * Right);
			Up = glm::normalize(qroll * Up);
		}

		InputManager::Instance(window).setMousePosition(windowWidth / 2, windowHeight / 2);

		AppendMessage("Orbit Camera - Target: (" + std::to_string(orbitTarget.x) + ", " + std::to_string(orbitTarget.y) + ", " + std::to_string(orbitTarget.z) + 
			") Position: (" + std::to_string(Position.x) + ", " + std::to_string(Position.y) + ", " + std::to_string(Position.z) + 
			") Distance: " +  std::to_string(orbitDistance));
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
	Forward = target - Position;
	updateOrientation();
}