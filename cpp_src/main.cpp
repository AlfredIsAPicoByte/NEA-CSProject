#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include "Engine.h"
#include "Mesh.h"
#include "colorClass.h"

int main(){
    if (InitGLFW() != 0) return -1;

	// Create the window and OpenGL context
	GLFWwindow* window = glfwCreateWindow(800, 800, "My OpenGL Window", nullptr, nullptr);
	if (!window) {
	    std::cerr << "Failed to create GLFW window" << std::endl;
	    glfwTerminate();
	    return -1;
	}

	// Make the context current
	glfwMakeContextCurrent(window);
	
    if (InitGLAD() != 0) return -1;

	// Load OpenGL functions with glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
	    std::cerr << "Failed to initialize GLAD" << std::endl;
	    return -1;
	}

	EnableOpenGLDebugger();

	// Main render loop
	MainLoop(window, [](GLFWwindow* window){
		Color clearColor("#4c5155ff");
        glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	    WaitForEscape(window);
	});

	// Clean up and exit
	CleanUp(window);

	return 0;
}
