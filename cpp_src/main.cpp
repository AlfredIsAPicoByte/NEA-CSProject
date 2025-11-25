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

	// Load model
	Shader bunnyShader("default.vert", "default.frag");
	Model bunny("bunny/scene.gltf");

	// Bunny properties
	glm::vec3 bunnyPos = glm::vec3(-5.0f, 5.0f, 0.0f);
	glm::quat bunnyRot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	glm::vec3 bunnyScale = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::mat4 bunnyModel = glm::mat4(1.0f);
	bunnyModel = glm::translate(bunnyModel, bunnyPos);
	bunnyModel *= glm::mat4_cast(bunnyRot);
	bunnyModel = glm::scale(bunnyModel, bunnyScale);

	bunnyShader.Activate();
	glUniformMatrix4fv(glGetUniformLocation(bunnyShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(bunnyModel));
	
	// Load model
	Shader swordShader("default.vert", "default.frag");
	Model sword("sword/scene.gltf");

	// Sword properties
	glm::vec3 swordPos = glm::vec3(5.0f, 0.0f, 0.0f);
	glm::quat swordRot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	glm::vec3 swordScale = glm::vec3(0.05f, 0.05f, 0.05f);
	glm::mat4 swordModel = glm::mat4(1.0f);
	swordModel = glm::translate(swordModel, swordPos);
	swordModel *= glm::mat4_cast(swordRot);
	swordModel = glm::scale(swordModel, swordScale);

	swordShader.Activate();
	glUniformMatrix4fv(glGetUniformLocation(swordShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(swordModel));

	// Enable depth (3D)
	glEnable(GL_DEPTH_TEST);

	Time time;
	Camera camera(windowWidth, windowHeight, glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	camera.fov = 60.0f;
	camera.speed = 0.5f;
	camera.speedFactor = 3.0f;

	int camMode = 1;
	bool resetKeyPressed = false;
	// Main loop
	Engine::Instance().Start();
	Engine::Instance().applyClearColor(bgClolor);
	Engine::Instance().Update(window,
		// Procecing and input
		[&]() {
			time.update();

			camera.Move(window, time.deltaTime);
			camera.updateMatrix();

			InputManager::Instance(window).doWhenKey(GLFW_KEY_1, false, true, [&]() {
				if (camMode != 1) {
					camera.mode = FIRST_PERSON;
					camMode = 1;
					AppendMessage("Set camera momvent mode to FIRST_PERSON");
				}
			});
			InputManager::Instance(window).doWhenKey(GLFW_KEY_2, false, true, [&]() {
				if (camMode != 2) {
					camera.mode = PLANE;
					camMode = 2;
					camera.SetPlaneTarget(objectPos);
					AppendMessage("Set camera momvent mode to PLANE");
				}
			});
			InputManager::Instance(window).doWhenKey(GLFW_KEY_3, false, true, [&]() {
				if (camMode != 3) {
					camera.mode = ORBIT;
					camMode = 3;
					camera.SelectOrbitTarget(lightPos);
					camera.orbitDistance = glm::length(camera.Position - lightPos);
					AppendMessage("Set camera momvent mode to ORBIT");
				}
			});

			InputManager::Instance(window).doWhenKey(GLFW_KEY_R, false, true, [&]() {
				if (!resetKeyPressed) {
					camera.Reset(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f));
					AppendMessage("Camera reset to default position and orientation");
					resetKeyPressed = true;
				}
			});
			InputManager::Instance(window).doWhenKey(GLFW_KEY_R, false, false, [&]() {
				resetKeyPressed = false;
			});
			InputManager::Instance(window).doWhenKey(GLFW_KEY_T, false, true, [&]() {
				if (!resetKeyPressed) {
					camera.ResetPlane();
					AppendMessage("Camera Plane mode reset to default orientation and power");
					resetKeyPressed = true;
				}
			});
			InputManager::Instance(window).doWhenKey(GLFW_KEY_T, false, false, [&]() {
				resetKeyPressed = false;
			});
			InputManager::Instance(window).doWhenKey(GLFW_KEY_Y, false, true, [&]() {
				if (!resetKeyPressed)  {
					camera.ResetOrbit();
					AppendMessage("Camera Orbit mode reset to default orientation and distance");
					resetKeyPressed = true;
				}
			});
			InputManager::Instance(window).doWhenKey(GLFW_KEY_Y, false, false, [&]() {
				resetKeyPressed = false;
			});

			InputManager::Instance(window).doWhenKey(GLFW_KEY_ESCAPE, false, true, [&]() {
				Engine::Instance().Exit();
			});
		},
		// Render
		[&]() {
			Engine::Instance().applyClearColor(bgClolor);

			// Draw objects
			floor.Draw(shaderProgram, camera);
			light.Draw(lightShader, camera);
			bunny.Draw(bunnyShader, camera);
			sword.Draw(swordShader, camera);
			
		},
		// ImGui Objects
		[&]() {
			ImGui::Begin("Info Panel");
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", time.deltaTime * 1000.0f, time.frameRate);
			ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera.Position.x, camera.Position.y, camera.Position.z);
			ImGui::Text("Camera Orientation: (%.2f, %.2f, %.2f)", camera.Forward.x, camera.Forward.y, camera.Forward.z);
			ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", camera.Up.x, camera.Up.y, camera.Up.z);
			ImGui::Text("Camera Mode: %s", camera.mode == FIRST_PERSON ? "FIRST_PERSON" : camera.mode == PLANE ? "PLANE" : "ORBIT");
			ImGui::Text("Controls:");
			ImGui::Text("  - WASD to move");
			ImGui::Text("  - QE to move up and down, or roll in Plane mode");
			ImGui::Text("  - Mouse to look around");
			ImGui::Text("  - Scroll to zoom (Orbit mode)");
			ImGui::Text("  - 1: First Person mode");
			ImGui::Text("  - 2: Plane mode");
			ImGui::Text("  - 3: Orbit mode");
			ImGui::Text("  - R: Reset camera");
			ImGui::Text("  - T: Reset Plane mode");
			ImGui::Text("  - Y: Reset Orbit mode");
			ImGui::End();

			// Todo: Add hiearchy for objects 
		});
		
	// Clean up and exit
	shaderProgram.Delete();
	lightShader.Delete();
	floor.CleanUp();
	light.CleanUp();
	bunny.CleanUp();
	sword.CleanUp();
	Engine::Instance().cleanUp(window);

	return 0;
}
