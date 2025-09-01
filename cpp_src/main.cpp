#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include "Engine.h"

int main(){
	// Initialize GLFW
	InitGLFW();

	// Create a GLFWwindow object of 800 by 800 pixels, naming it "YoutubeOpenGL"
	GLFWwindow* window = createWindow(800, 800, "YoutubeOpenGL");
	if (window == nullptr) return -1;

	// Load GLAD so it configures OpenGL
	InitGLAD();

	// Setup OpenGL debug context
	SetupOpenGLDebug();

	// Main render loop
	MainLoop(window, [](GLFWwindow* window){
	}, Color("#7999bdff"));

	// Clean up and exit
	CleanUp(window);

	return 0;
}
