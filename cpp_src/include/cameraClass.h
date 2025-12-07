#pragma once

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
#include "Debugger.h"
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
	glm::vec3 WorldUp;
	glm::vec3 Up, Forward, Right;
	glm::mat4 viewMatrix = glm::mat4(1.0f);
	glm::mat4 projectionMatrix = glm::mat4(1.0f);
	glm::mat4 cameraMatrix = glm::mat4(1.0f);

	// Prevents the camera from jumping around when first clicking left click
	bool firstClick = true;

	// Stores the width and height of the window
	int windowWidth;
	int windowHeight;
	float aspectRatio;

	std::string name = "";

	CameraType type = PERSPECTIVE;
	CamaraMovementMode mode = FIRST_PERSON;

	float fov = 45.0f;
	float nearPlane = 0.1f;
	float farPlane = 100.0f;


	// Adjust the speed of the camera and it's sensitivity when looking around
	float speed = 0.1f;
	float speedFactor = 4.0f;
	float mouseSensitivity = 100.0f;
	bool invertY = false;
	bool invertX = false;

	// Plane mode state
	float planeRotateSpeed = 5.0f;
	float planePitch = 0.0f;
	float planeYaw = -90.0f;
	float planeRoll = 0.0f;
	float planePower = 0.0f;
	float planeMaxPower = 5.0f;
	float planeMinPower = -5.0f;
	float planePowerIncrement = 1.0f;
	bool isPlanePowerIncreasing = false;
	bool isPlaneActivelyStoping = false;

	// Orbit mode state
	float orbitSpeed = 10.0f;
	float orbitPitch = 0.0f;
	float orbitYaw = 0.0f;
	float orbitRoll = 0.0f;
	float orbitDistance = 5.0f;
	float orbitMinDistance = 0.1f;
	float orbitMaxDistance = 100.0f;
	float orbitZoomSpeed = 0.1f;
	glm::vec3 orbitTarget = glm::vec3(0.0f);

	// Camera constructor to set up initial values
	Camera(int width, int height, glm::vec3 position, glm::vec3 forward, glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f));

	// Updates the camera matrix to the Vertex Shader
	void updateMatrix();

	// Resize the camera aspect ratio
	void Resize(int width, int height);

	// Exports the camera matrix to a shader
	void SetModelMatrixUniform(Shader& shader, const char* uniform);

	void Move(GLFWwindow* window, double deltaTime);

	void SetPlaneTarget(glm::vec3 target);
	void SelectOrbitTarget(const glm::vec3& target);
	void LookAt(const glm::vec3& target);

	// scroll callback that can access Camera members
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

	void ResetOrbit();
	void ResetPlane();
	void Reset(glm::vec3 position, glm::vec3 forward, glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f));
private:
	int id;
	
	void FirstPersonMovement(GLFWwindow* window, double deltaTime);
	void PlaneMovement(GLFWwindow* window, double deltaTime);
	void OrbitMovement(GLFWwindow* window, double deltaTime);
};