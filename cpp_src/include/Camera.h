#ifndef CAMERA_CLASS_H
#define CAMERA_CLASS_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <string>

#include "shaderClass.h"
#include "timeClass.h"
#include "Debug.h"
#include "InputManager.h"

enum CameraType {
	PERSPECTIVE,
	ORTHOGRAPHIC
};

enum CamaraMovementMode {
	FIRST_PERSON,
	PLANE,
	ORBIT
};

class Camera
{
public:
	// Stores the main vectors of the camera
	glm::vec3 Position;
	glm::vec3 Orientation = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 Up;
	glm::vec3 WorldUp;
	glm::vec3 Up, Forward, Right;
	glm::mat4 viewMatrix = glm::mat4(1.0f),
		projectionMatrix = glm::mat4(1.0f),
		cameraMatrix = glm::mat4(1.0f);

	// Prevents the camera from jumping around when first clicking left click
	bool firstClick = true,
		cursorHidden = false;

	// Stores the width and height of the window
	int windowWidth, windowHeight;
	float aspectRatio;

	std::string name = "";
	int id;

	CameraType type = PERSPECTIVE;
	CamaraMovementMode mode = FIRST_PERSON;

	float fov = 45.0f,
		nearPlane = 0.1f,
		farPlane = 100.0f;


	// Adjust the speed of the camera and it's sensitivity when looking around
	float speed = 0.1f,
		speedFactor = 4.0f,
		mouseSensitivity = 100.0f;
	bool invertY = false;
	bool invertX = false;

	// Plane mode state
	float planeRotateSpeed = 5.0f,
		planePitch = 0.0f,
		planeYaw = -90.0f,
		planeRoll = 0.0f,
		planePower = 0.0f,
		planeMaxPower = 5.0f,
		planeMinPower = -5.0f,
		planePowerIncrement = 1.0f;
	bool isPlanePowerIncreasing = false;
	bool isPlaneActivelyStoping = false;

	// Orbit mode state
	float orbitSpeed = 10.0f,
		orbitPitch = 0.0f,
		orbitYaw = 0.0f,
		orbitRoll = 0.0f,
		orbitRollSpeed = glm::degrees(0.05f),
		orbitDistance = 5.0f,
		orbitMinDistance = 0.1f,
		orbitMaxDistance = 100.0f,
		orbitZoomSpeed = 0.1f;
	glm::vec3 orbitTarget = glm::vec3(0.0f);

	// Camera constructor to set up initial values
	Camera(int width, int height, glm::vec3 position);

	// Updates the camera matrix to the Vertex Shader
	void updateMatrix();
	void updateOrientation();

	// Resize the camera aspect ratio
	void Resize(int width, int height);

	// Exports the camera matrix to a shader
	void Matrix(Shader& shader, const char* uniform);

	void Move(GLFWwindow* window, double deltaTime);

	void SetPlaneTarget(glm::vec3 target);
	void SelectOrbitTarget(const glm::vec3& target);
	void LookAt(const glm::vec3& target);

	// scroll callback that can access Camera members
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
private:
	void FirstPersonMovement(GLFWwindow* window, double deltaTime);
	void PlaneMovement(GLFWwindow* window, double deltaTime);
	void OrbitMovement(GLFWwindow* window, double deltaTime);
};
#endif