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
	glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::mat4 viewMatrix = glm::mat4(1.0f);
	glm::mat4 projectionMatrix = glm::mat4(1.0f);
	glm::mat4 cameraMatrix = glm::mat4(1.0f);

	// Prevents the camera from jumping around when first clicking left click
	bool firstClick = true;

	// Stores the width and height of the window
	int windowWidth;
	int windowHeight;
	float aspectRatio;

	CameraType type;
	CamaraMovementMode mode;

	// Adjust the speed of the camera and it's sensitivity when looking around
	float moveSpeed = 0.1f;
	float speedMult = 4.0f;
	float sensitivity = 100.0f;

	float fov = 45.0f;
	float nearPlane = 0.1f;
	float farPlane = 100.0f;

	std::string name = "";
	int id;

	// Orbit mode state
	float orbitPitch = 0.0f;
	float orbitYaw = 0.0f;
	float orbitDistance = 5.0f;
	float orbitMinDistance = 0.1f;
	float orbitMaxDistance = 100.0f;
	float orbitZoomSpeed = 0.1f;
	glm::vec3 orbitTarget = glm::vec3(0.0f);

	// Camera constructor to set up initial values
	Camera(int width, int height, glm::vec3 position);

	// Updates the camera matrix to the Vertex Shader
	void updateMatrix();
	// Exports the camera matrix to a shader
	void Matrix(Shader& shader, const char* uniform);

	void Move(GLFWwindow* window, Time& time);

	void SelectOrbitTarget(const glm::vec3& target);
	void LookAt(const glm::vec3& target);

	// scroll callback that can access Camera members
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
private:
	void FirstPersonMovement(GLFWwindow* window, Time& time);
	void PlaneMovement(GLFWwindow* window, Time& time);
	void OrbitMovement(GLFWwindow* window, Time& time);
};
#endif