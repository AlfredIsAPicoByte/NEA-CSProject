#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include "Engine.h"
#include "Mesh.h"
#include "colorClass.h"

int main(){
	// Initialize GLFW
	if (!glfwInit()) {
	    std::cerr << "Failed to initialize GLFW" << std::endl;
	    return -1;
	}

	// Set GLFW window hints (optional, for OpenGL version/profile)
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create the window and OpenGL context
	GLFWwindow* window = glfwCreateWindow(800, 800, "My OpenGL Window", nullptr, nullptr);
	if (!window) {
	    std::cerr << "Failed to create GLFW window" << std::endl;
	    glfwTerminate();
	    return -1;
	}

	// Make the context current
	glfwMakeContextCurrent(window);

	// Load OpenGL functions with glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
	    std::cerr << "Failed to initialize GLAD" << std::endl;
	    return -1;
	}

	// Main render loop
	MainLoop(window, [](GLFWwindow* window){
		Color clearColor("#575c61ff");
        glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	    WaitForEscape(window);
	});

	// Clean up and exit
	CleanUp(window);

	return 0;
}
