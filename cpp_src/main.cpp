#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include "Engine.h"

const unsigned int width = 800;
const unsigned int height = 800;

Vertex vertices[] =
{ //               COORDINATES           /            COLORS          /           NORMALS         /       TEXTURE COORDINATES    //
	Vertex{glm::vec3(-1.0f, 0.0f,  1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(0.0f, 0.0f)},
	Vertex{glm::vec3(-1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(0.0f, 1.0f)},
	Vertex{glm::vec3( 1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(1.0f, 1.0f)},
	Vertex{glm::vec3( 1.0f, 0.0f,  1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(1.0f, 0.0f)}
};

GLuint indices[] =
{
	0, 1, 2,
	0, 2, 3
};

Vertex lightVertices[] =
{ //     COORDINATES     //
	Vertex{glm::vec3(-0.1f, -0.1f,  0.1f)},
	Vertex{glm::vec3(-0.1f, -0.1f, -0.1f)},
	Vertex{glm::vec3(0.1f, -0.1f, -0.1f)},
	Vertex{glm::vec3(0.1f, -0.1f,  0.1f)},
	Vertex{glm::vec3(-0.1f,  0.1f,  0.1f)},
	Vertex{glm::vec3(-0.1f,  0.1f, -0.1f)},
	Vertex{glm::vec3(0.1f,  0.1f, -0.1f)},
	Vertex{glm::vec3(0.1f,  0.1f,  0.1f)}
};

GLuint lightIndices[] =
{
	0, 1, 2,
	0, 2, 3,
	0, 4, 7,
	0, 7, 3,
	3, 7, 6,
	3, 6, 2,
	2, 6, 5,
	2, 5, 1,
	1, 5, 4,
	1, 4, 0,
	4, 5, 6,
	4, 6, 7
};

int main(){
    if (InitGLFW() != 0) return -1;

	// Create the window and OpenGL context
	GLFWwindow* window = glfwCreateWindow(width, height, "My OpenGL Window", nullptr, nullptr);
	if (!window) {
	    AppendError("[Main] Failed to create GLFW window");
	    glfwTerminate();
	    return -1;
	}

	// Make the context current
	glfwMakeContextCurrent(window);
	
    if (InitGLAD() != 0) return -1;

	// Load OpenGL functions with glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
	    AppendError("[Main] Failed to initialize GLAD");
	    return -1;
	}

#if DEBUG_MODE
	EnableOpenGLDebugger();
	std::cout << "Debugger On!" << std::endl;
#else
    std::cout << "Debugger Off!" << std::endl;
#endif

	Texture textures[] {
		Texture("planks.png", "diffuse", 0, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE),
		Texture("planksSpec.png", "specular", 1, GL_R8, GL_RED, GL_UNSIGNED_BYTE)
	};

	glEnable(GL_DEPTH_TEST);

	Color bgClolor("#454749ff");
	Camera camera(width, height, glm::vec3(0.0f, 0.0f, 2.0f));

	// Main render loop
	MainLoop(window, [&](GLFWwindow* window){
		applyClearColor(bgClolor);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



		WaitForEscape(window);
	});

	// Clean up and exit
	CleanUp(window);

	return 0;
}
