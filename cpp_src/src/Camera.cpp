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

void Camera::Move(GLFWwindow* window, Time& time)
{
	switch (mode) {
		case FIRST_PERSON:
			FirstPersonMovement(window, time);
			break;
		case PLANE:
			PlaneMovement(window, time);
			break;
		case ORBIT:
			OrbitMovement(window, time);
			break;
		default:
			// Unknown mode
			break;
	}
}

void Camera::FirstPersonMovement(GLFWwindow* window, Time& time)
{
    float _speed = speed * time.deltaTime; // Movement speed

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
		_speed = speed * speedFactor * time.deltaTime;
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
	{
		_speed = speed * time.deltaTime;
	}


	// Handles mouse inputs
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
	{
		// Hides mouse cursor
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

		// Prevents camera from jumping on the first click
		if (firstClick)
		{
			glfwSetCursorPos(window, (windowWidth / 2), (windowHeight / 2));
			firstClick = false;
		}

		// Stores the coordinates of the cursor
		double mouseX;
		double mouseY;
		// Fetches the coordinates of the cursor
		glfwGetCursorPos(window, &mouseX, &mouseY);

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

		// Rotates the Orientation left and right
		Orientation = glm::rotate(Orientation, glm::radians(-rotY), Up);

		// Sets mouse cursor to the middle of the screen so that it doesn't end up roaming around
		glfwSetCursorPos(window, (windowWidth / 2), (windowHeight / 2));
	}
	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE)
	{
		// Unhides cursor since camera is not looking around anymore
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		// Makes sure the next time the camera looks around it doesn't jump
		firstClick = true;
	}
}

void Camera::SetPlaneTarget(glm::vec3 target)
{
	planePitch = glm::degrees(asin(target.y - Position.y));
	planeYaw = glm::degrees(atan2(target.x - Position.x, target.z - Position.z));
	planeRoll = 0.0f;
}

void Camera::PlaneMovement(GLFWwindow* window, Time& time)
{
	float rotationSpeed = planeRotateSpeed * time.deltaTime; // Rotation speed

	// Handles key inputs
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		planePitch -= rotationSpeed * 10.0f * (invertY ? -1.0f : 1.0f);
		if (planePitch < -89.0f) planePitch = -89.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		planePitch += rotationSpeed * 10.0f * (invertY ? -1.0f : 1.0f);
		if (planePitch > 89.0f) planePitch = 89.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		planeYaw += rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f);
		if (planeYaw > 180.0f) planeYaw -= 360.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		planeYaw -= rotationSpeed * 10.0f * (invertX ? -1.0f : 1.0f);
		if (planeYaw < -180.0f) planeYaw += 360.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		planeRoll -= rotationSpeed * 10.0f;
		if (planeRoll < -180.0f) planeRoll += 360.0f;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		planeRoll += rotationSpeed * 10.0f;
		if (planeRoll > 180.0f) planeRoll -= 360.0f;
	}

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
	{
		planePower += planePowerIncrement * time.deltaTime;
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
	{
		planePower -= planePowerIncrement * time.deltaTime;
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
	{
		if (planePower > 0) {
			planePower -= planePowerIncrement * time.deltaTime;
		}
		else if (planePower < 0) {
			planePower += planePowerIncrement * time.deltaTime;
		}
		
		if (abs(planePower) < 0.01f) {
			planePower = 0.0f;
		}
	}
	
	if (planePower > planeMaxPower) {
		planePower = planeMaxPower;
	}
	else if (planePower < planeMinPower) {
		planePower = planeMinPower;
	}

	// Update Orientation based on pitch, yaw, and roll
	glm::vec3 front;
	// Build a quaternion from Euler angles (pitch -> X, yaw -> Y, roll -> Z)
	glm::quat q = glm::quat(glm::vec3(glm::radians(planePitch), glm::radians(planeYaw), glm::radians(planeRoll)));

	front.x = q.x * q.w * 2.0f + q.y * q.z * 2.0f;
	front.y = q.y * q.w * 2.0f - q.x * q.z * 2.0f;
	front.z = 1.0f - (q.x * q.x * 2.0f + q.y * q.y * 2.0f);

	// Apply roll to Up as well so lookAt will reflect roll tilt
	Up = q * WorldUp; // Use the stored world up vector as the base
	Orientation = glm::normalize(front);
	
	// Move the camera forward in the direction it's facing
	Position += planePower * Orientation * time.deltaTime;
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

void Camera::OrbitMovement(GLFWwindow* window, Time& time)
{
    // ensure the window user pointer points to this Camera instance
    glfwSetWindowUserPointer(window, this);
    // set the scroll callback to the static Camera method
    glfwSetScrollCallback(window, Camera::scroll_callback);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		if (firstClick) {
			glfwSetCursorPos(window, windowWidth / 2, windowHeight / 2);
			firstClick = false;
		}

		double mouseX, mouseY;
		glfwGetCursorPos(window, &mouseX, &mouseY);

		float mouse_dx = (float)(mouseX - (windowWidth / 2));
		float mouse_dy = (float)(mouseY - (windowHeight / 2));

		float orbit_speed = mouseSensitivity * time.deltaTime;
		
		float yaw = orbit_speed * mouse_dx / windowWidth;

		float pitch = orbit_speed * mouse_dy / windowHeight;
		float max_pitch = glm::radians(89.0f);
		float new_pitch = orbitPitch + pitch;

		if (abs(new_pitch) > max_pitch) {
			pitch = max_pitch * (new_pitch > 0 ? 1 : -1) - orbitPitch;
		}

		orbitPitch += pitch * (invertY ? -1.0f : 1.0f);
		orbitYaw += yaw * (invertX ? -1.0f : 1.0f);
		
		// Clamp radius
		if (orbitDistance < orbitMinDistance) orbitDistance = orbitMinDistance;
		if (orbitDistance > orbitMaxDistance) orbitDistance = orbitMaxDistance;

		// Spherical coordinates
		float x = orbitTarget.x + orbitDistance * cos(orbitPitch) * sin(orbitYaw);
		float y = orbitTarget.y + orbitDistance * sin(orbitPitch);
		float z = orbitTarget.z + orbitDistance * cos(orbitPitch) * cos(orbitYaw);

		Position = glm::vec3(x, y, z);

		LookAt(orbitTarget);
		
		glfwSetCursorPos(window, windowWidth / 2, windowHeight / 2);
	} else {
		firstClick = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void Camera::LookAt(const glm::vec3& target)
{
	Orientation = glm::normalize(target - Position);
}