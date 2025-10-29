#include "Camera.h"

Camera::Camera(int width, int height, glm::vec3 position)
{
	windowWidth = width;
	windowHeight = height;
    aspectRatio = static_cast<float>(windowWidth / windowHeight);
    Position = position;
}

void Camera::updateMatrix()
{
	// Initializes matrices since otherwise they will be the null matrix
	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 projection = glm::mat4(1.0f);

	// Makes camera look in the right direction from the right position
	view = glm::lookAt(Position, Position + Orientation, Up);
	// Adds perspective to the scene
	projection = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);

	// Sets new camera matrix
	cameraMatrix = projection * view;
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
    float speed = moveSpeed * time.deltaTime; // Movement speed

	// Handles key inputs
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		Position += speed * Orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		Position += speed * -glm::normalize(glm::cross(Orientation, Up));
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		Position += speed * -Orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		Position += speed * glm::normalize(glm::cross(Orientation, Up));
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		Position += speed * Up;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		Position += speed * -Up;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
	{
		speed = moveSpeed * speedMult * time.deltaTime;
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
	{
		speed = moveSpeed * time.deltaTime;
	}


	// Handles mouse inputs
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
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
		float rotX = sensitivity * (float)(mouseY - (windowWidth / 2)) / windowWidth;
		float rotY = sensitivity * (float)(mouseX - (windowHeight / 2)) / windowHeight;

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
	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
	{
		// Unhides cursor since camera is not looking around anymore
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		// Makes sure the next time the camera looks around it doesn't jump
		firstClick = true;
	}
}

void Camera::PlaneMovement(GLFWwindow* window, Time& time)
{
	float velocity = moveSpeed * time.deltaTime;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		velocity *= speedMult;
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		Position += Orientation * velocity;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		Position -= Orientation * velocity;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		Position -= glm::normalize(glm::cross(Orientation, Up)) * velocity;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		Position += glm::normalize(glm::cross(Orientation, Up)) * velocity;
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		Position += Up * velocity;
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		Position -= Up * velocity;

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
		float rot_x = (sensitivity * mouse_dx / windowWidth) * time.deltaTime;
		float rot_y = (sensitivity * mouse_dy / windowHeight) * time.deltaTime;
		if (abs(rot_y) > 89.0f) {
			rot_y = 89.0f * (rot_y > 0 ? 1 : -1);
		}
		// Yaw rotation around up
		Orientation = glm::rotate(Orientation, -glm::radians(rot_x), Up);
		// Pitch rotation around right
		glm::vec3 right = glm::normalize(glm::cross(Orientation, Up));
		Orientation = glm::rotate(Orientation, -glm::radians(rot_y), right);
		glfwSetCursorPos(window, windowWidth / 2, windowHeight / 2);
	} else {
		firstClick = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
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

		float orbit_speed = sensitivity * time.deltaTime;
		
		float yaw = orbit_speed * mouse_dx / windowWidth;

		float pitch = orbit_speed * mouse_dy / windowHeight;
		float max_pitch = glm::radians(89.0f);
		float new_pitch = orbitPitch + pitch;

		if (abs(new_pitch) > max_pitch) {
			pitch = max_pitch * (new_pitch > 0 ? 1 : -1) - orbitPitch;
		}

		orbitPitch += pitch;
		orbitYaw += yaw;
		orbitDistance = glm::length(Position - orbitTarget);
		
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