#include "Engine.h"

int windowWidth = 800;
int windowHeight = 800;
std::string windowTitle = "My OpenGL Window";
Color bgClolor("#454749ff");

// Vertices coordinates
Vertex vertices[] =
{ //               COORDINATES           /            COLORS          /           NORMALS         /       TEXTURE COORDINATES    //
	Vertex{glm::vec3( 1.0f, 0.0f,  1.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(0.0f, 0.0f)},
	Vertex{glm::vec3( 1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(0.0f, 1.0f)},
	Vertex{glm::vec3(-1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(1.0f, 1.0f)},
	Vertex{glm::vec3(-1.0f, 0.0f,  1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec2(1.0f, 0.0f)}
};

// Indices for vertices order
GLuint indices[] =
{
	0, 1, 3,
	1, 2, 3
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

int main()
{
	AppendMessage("Starting Engine...");
	
    // Initialize GLFW
    if (!glfwInit()) {
		AppendOpenGLError("Failed to initialize GLFW");
		return -1;
	}

    // Tell GLFW what version of OpenGL we are using 
    // Example: OpenGL 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Enable debug context (optional)
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);


	// Create the window and OpenGL context
	// Create a GLFWwindow object
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, windowTitle.c_str(), nullptr, nullptr);
    
	// Error check if the window fails to create
    if (window == NULL || window == nullptr)
    {
		AppendOpenGLError("Failed to create GLFW window");
        glfwTerminate();
        return -1;
    }

	// Make the context current
	glfwMakeContextCurrent(window);

	// Load OpenGL functions with glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		AppendOpenGLError("Failed to load OpenGL Functions with GLAD");
	    return -1;
	}

    // Specify the viewport of OpenGL in the Window
    // In this case the viewport goes from x = 0, y = 0, to x = width, y = height
    glViewport(0, 0, windowWidth, windowHeight);

#if DEBUG_MODE
	EnableOpenGLDebugger();
	std::cout << "Debugger On!" << std::endl;
#else
    std::cout << "Debugger Off!" << std::endl;
#endif

	Texture textures[] = {
		Texture("planks.png", "diffuse", 0),
		Texture("planks_specular.png", "specular", 1)
	};

	// Generates Shader object using shaders default.vert and default.frag
	Shader shaderProgram("default.vert", "default.frag");
	// Store mesh data in vectors for the mesh
	std::vector <Vertex> verts(vertices, vertices + sizeof(vertices) / sizeof(Vertex));
	std::vector <GLuint> ind(indices, indices + sizeof(indices) / sizeof(GLuint));
	std::vector <Texture> tex(textures, textures + sizeof(textures) / sizeof(Texture));
	// Create floor mesh
	Mesh floor(verts, ind, tex);


	// Shader for light cube
	Shader lightShader("light.vert", "light.frag");
	// Store mesh data in vectors for the mesh
	std::vector <Vertex> lightVerts(lightVertices, lightVertices + sizeof(lightVertices) / sizeof(Vertex));
	std::vector <GLuint> lightInd(lightIndices, lightIndices + sizeof(lightIndices) / sizeof(GLuint));
	// Create light mesh
	Mesh light(lightVerts, lightInd, tex);

	
	// Floor properties
	glm::vec3 objectPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::mat4 objectModel = glm::mat4(1.0f);
	objectModel = glm::translate(objectModel, objectPos);

	// Light properties
	glm::vec4 lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	glm::vec3 lightPos = glm::vec3(0.5f, 0.5f, 0.5f);
	glm::mat4 lightModel = glm::mat4(1.0f);
	lightModel = glm::translate(lightModel, lightPos);

	lightShader.Activate();
	glUniformMatrix4fv(glGetUniformLocation(lightShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(lightModel));
	glUniform4f(glGetUniformLocation(lightShader.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	shaderProgram.Activate();
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(objectModel));
	glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);

	PythonManager pm;
	pm.Initialize();
	pm.AddModulePath("C:/Users/atang/Documents/GitHub/Programing_Projects/NEA-CSProject/py_src/src");
	py::object primaryStructures = pm.LoadModule("PrimaryStructures");

	py::object ray = primaryStructures.attr("Ray")(py::make_tuple(0.0f, 1.0f, 0.0f), py::make_tuple(1.0f, -1.0f, 0.0f));
	AppendPythonMessage("Created Ray from Python module PrimaryStructures" + static_cast<std::string>(py::str(ray)));

	// Enable depth (3D)
	glEnable(GL_DEPTH_TEST);

	Time timer;
	Camera camera(windowWidth, windowHeight, glm::vec3(0.0f, 0.0f, 2.0f));
	camera.fov = 60.0f;
	camera.speed = 0.5f;
	camera.speedFactor = 3.0f;

	int camMode = 1;
	// Main loop
	Engine::Instance().Start();
	Engine::Instance().applyClearColor(bgClolor);
	Engine::Instance().Update(window,
		// Render
		[&]() {
			timer.update();
			Engine::Instance().applyClearColor(bgClolor);

			camera.Move(window, timer);
			camera.updateMatrix();

			// Draw floor
			floor.Draw(shaderProgram, camera);
			light.Draw(lightShader, camera);
			
			if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS & camMode != 1) {
				camera.mode = FIRST_PERSON;
				camMode = 1;
				AppendMessage("Set camera momvent mode to FIRST_PERSON");
			}
			else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS & camMode != 2) {
				camera.mode = PLANE;
				camMode = 2;
				camera.SetPlaneTarget(-objectPos);
				AppendMessage("Set camera momvent mode to PLANE");
			}
			else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS & camMode != 3) {
				camera.mode = ORBIT;
				camMode = 3;
				camera.SelectOrbitTarget(lightPos);
				AppendMessage("Set camera momvent mode to ORBIT");
			}
			
		}
	);

	// Clean up and exit
	Engine::Instance().cleanUp(window);

	return 0;
}
