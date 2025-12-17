#include "Engine.h"

int windowWidth = 800;
int windowHeight = 800;
std::string windowTitle = "My OpenGL Window";
Color bgClolor("#454749ff");

// Vertices coordinates
Vertex vertices[] =
{ //               POSITION             /       	 NORMAL          /            COLOR           /       TEXCOORDS      //
    Vertex{glm::vec3( 1.0f, 0.0f,  1.0f), glm::vec3(0.0f, 1.0f, 0.0f), Color("#fff").toVec3(), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3( 1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), Color("#fff").toVec3(), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(-1.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), Color("#fff").toVec3(), glm::vec2(1.0f, 1.0f)},
    Vertex{glm::vec3(-1.0f, 0.0f,  1.0f), glm::vec3(0.0f, 1.0f, 0.0f), Color("#fff").toVec3(), glm::vec2(1.0f, 0.0f)}
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
		AppendGraphicsError("Failed to initialize GLFW");
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
		AppendGraphicsError("Failed to create GLFW window");
        glfwTerminate();
        return -1;
    }

	// Make the context current
	glfwMakeContextCurrent(window);

	// Load OpenGL functions with glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		AppendGraphicsError("Failed to load OpenGL Functions with GLAD");
	    return -1;
	}

    // Specify the viewport of OpenGL in the Window
    // In this case the viewport goes from x = 0, y = 0, to x = width, y = height
    glViewport(0, 0, windowWidth, windowHeight);

	glfwSetWindowTitle(window, "A level NEA Rendering Engine");

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

	// Create shader programs
	Shader litShaderProgram("lit.vert", "lit.frag");
	Shader materialProgram("material_lit.vert", "material_lit.frag");
	Shader unlitShaderProgram("unlit.vert", "unlit.frag");

	Material roughtMat(Color("#2cc20eff").toVec3(), 0.0f, 0.52f, 0.0f, 0.0f, 0.0f, Color("#000").toVec3());

	std::vector<Vertex> floorVerts(vertices, vertices + sizeof(vertices) / sizeof(vertices[0]));
	std::vector<GLuint> floorInd(indices, indices + sizeof(indices) / sizeof(indices[0]));
	std::vector<Texture> floorTex(textures, textures + sizeof(textures) / sizeof(textures[0]));
	Mesh floorMesh(floorVerts, floorInd, floorTex);
	floorMesh.SetName("Floor");
	glm::vec3 objectPos = glm::vec3(0.0f, -0.5f, 0.0f);
	floorMesh.SetModelMatrix(glm::translate(floorMesh.GetModelMatrix(), objectPos));

	std::vector<Vertex> lightVerts(lightVertices, lightVertices + sizeof(lightVertices) / sizeof(lightVertices[0]));
	std::vector<GLuint> lightInd(lightIndices, lightIndices + sizeof(lightIndices) / sizeof(lightIndices[0]));
	std::vector<Texture> lightTex;
	Mesh lightMesh(lightVerts, lightInd, lightTex);
	lightMesh.SetName("Light");
	glm::vec3 lightPos = glm::vec3(0.5f, 0.5f, 0.5f);
	lightMesh.SetModelMatrix(glm::translate(lightMesh.GetModelMatrix(), lightPos));
	Color lightColor("#f8ffcfff");

	auto bunnyPtr = std::make_shared<Model>("bunny/scene.gltf");
	glm::vec3 bunnyPos = glm::vec3(0.0f, -0.5f, 0.0f);
	glm::vec3 bunnyRot = glm::vec3(90.0f, -90.0f, 50.0f);
	glm::vec3 bunnyScale = glm::vec3(2.0f);
	glm::mat4 bunnyModelMatrix = glm::mat4(1.0f);
	bunnyModelMatrix = glm::translate(bunnyModelMatrix, bunnyPos);
	bunnyModelMatrix = glm::rotate(bunnyModelMatrix, glm::radians(bunnyRot.y), glm::vec3(0.0f, 1.0f, 0.0f));
	bunnyModelMatrix = glm::rotate(bunnyModelMatrix, glm::radians(bunnyRot.x), glm::vec3(1.0f, 0.0f, 0.0f));
	bunnyModelMatrix = glm::rotate(bunnyModelMatrix, glm::radians(bunnyRot.z), glm::vec3(0.0f, 0.0f, 1.0f));	
	bunnyModelMatrix = glm::scale(bunnyModelMatrix, bunnyScale);
	bunnyPtr->SetModelMatricesForAllMeshes(std::vector<glm::mat4>{ bunnyModelMatrix });
	auto bunny = std::make_shared<ModelMeshAdapter>(bunnyPtr, 0);
	bunny->SetName("Bunny");

	auto swordPtr = std::make_shared<Model>("sword/scene.gltf");
	glm::vec3 swordPos = glm::vec3(0.0f, 0.0f, -2.0f);
	glm::vec3 swordScale = glm::vec3(0.05f);
	glm::mat4 swordModelMatrix = glm::mat4(1.0f);
	swordModelMatrix = glm::translate(swordModelMatrix, swordPos);
	swordModelMatrix = glm::scale(swordModelMatrix, swordScale);
	swordPtr->SetModelMatricesForAllMeshes(std::vector<glm::mat4>{ swordModelMatrix });
	auto sword = std::make_shared<ModelMeshAdapter>(swordPtr, 0);
	sword->SetName("Sword");

	Time time;
	Camera camera(windowWidth, windowHeight, glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	camera.fov = 60.0f;
	camera.speed = 0.5f;
	camera.speedFactor = 3.0f;

	litShaderProgram.Activate();
	litShaderProgram.setVec3("u_lightPos", lightPos);
	litShaderProgram.setVec3("u_lightColor", lightColor.toVec3());
	litShaderProgram.setVec3("u_viewPos", camera.Position);
	litShaderProgram.setVec2("u_lightRadius", glm::vec2(10.0f));
	litShaderProgram.setInt("u_lightType", 0);
	litShaderProgram.setFloat("u_specularStrength", 1.0f);
	litShaderProgram.setFloat("u_ambient", 0.05f);

	materialProgram.Activate();
	CreateLightsUBO();
	CreateMaterialUBO();
	UpdateLightsUBO(std::vector<Light>{ Light{lightPos, lightColor.toVec3(), 1.0f} });
	UpdateMaterialUBO(roughtMat);
	camera.SetModelMatrixUniform(materialProgram, "u_camMatrix");
	litShaderProgram.setVec3("u_viewDir", camera.Forward);

	unlitShaderProgram.Activate();
	camera.SetModelMatrixUniform(unlitShaderProgram, "u_camMatrix");
	unlitShaderProgram.setVec4("lightColor", lightColor.toVec4());

	int camMode = 1;
	bool resetKeyPressed = false;

	Scene testScene;
	testScene.SetCamera(&camera);
	testScene.Initialize();
	testScene.SetOpenGLRenderFunction(std::make_shared<std::function<void()>>([&]() {
		// update shader camera uniforms every frame
		litShaderProgram.Activate();
		camera.SetModelMatrixUniform(litShaderProgram, "u_camMatrix");
		litShaderProgram.setVec3("u_viewPos", camera.Position);
		
		materialProgram.Activate();
		UpdateLightsUBO(std::vector<Light>{ Light{lightPos, lightColor.toVec3(), 1.0f} });
		UpdateMaterialUBO(roughtMat);
		camera.SetModelMatrixUniform(materialProgram, "u_camMatrix");
		litShaderProgram.setVec3("u_viewDir", camera.Forward);

		unlitShaderProgram.Activate();
		camera.SetModelMatrixUniform(unlitShaderProgram, "u_camMatrix");

		// Draw objects
		floorMesh.Draw(litShaderProgram, camera);
		bunny->Draw(materialProgram, camera);
		sword->Draw(litShaderProgram, camera);
		lightMesh.Draw(unlitShaderProgram, camera);
	}));
	testScene.AddRenderable(std::make_shared<Mesh>(floorMesh));
	testScene.AddRenderable(std::make_shared<Mesh>(lightMesh));

	testScene.AddRenderable(bunny);
	testScene.AddRenderable(sword);

	testScene.selectedRenderable = 0;
	testScene.renderSettings->usePythonRendering = false;
	
	testScene.SaveScene("scenes/test_save.scn");

	// Main loop
	Engine::Instance().Start();
	Engine::Instance().applyClearColor(bgClolor);
	Engine::Instance().setDepthTest(true);
	Engine::Instance().Update(window,
		// PreProcecing 
		[&]() {
			time.update();

			std::string newTitle = "A level NEA Rendering Engine - (" + testScene.name + ") - " + std::to_string(time.frameRate) + "FPS, " + std::to_string(time.deltaTime * 1000.0f) + "ms";
			glfwSetWindowTitle(window, newTitle.c_str());
		},
		// Input handling
		[&]() {
			camera.Move(window, time.deltaTime);
			camera.updateMatrix();
			{
				InputManager::Instance(window).doWhenKey(GLFW_KEY_1, true, [&]() {
					if (camMode != 1) {
						camera.mode = FIRST_PERSON;
						camMode = 1;
						AppendMessage("Set camera momvent mode to FIRST_PERSON");
					}
				});
				InputManager::Instance(window).doWhenKey(GLFW_KEY_2, true, [&]() {
					if (camMode != 2) {
						camera.mode = PLANE;
						camMode = 2;
						camera.SetPlaneTarget(objectPos);
						AppendMessage("Set camera momvent mode to PLANE");
					}
				});
				InputManager::Instance(window).doWhenKey(GLFW_KEY_3, true, [&]() {
					if (camMode != 3) {
						camera.mode = ORBIT;
						camMode = 3;
						if (testScene.selectedRenderable > -1) {
							auto selected = testScene.renderables[testScene.selectedRenderable];
							// TODO: get selected pos
						} else {
							camera.SelectOrbitTarget(glm::vec3(0.0f));
							camera.orbitDistance = 4.0f;
						}
						AppendMessage("Set camera momvent mode to ORBIT");
					}
				});

				InputManager::Instance(window).doWhenKey(GLFW_KEY_R, true, [&]() {
					if (!resetKeyPressed) {
						camera.Reset(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f));
						AppendMessage("Camera reset to default position and orientation");
						resetKeyPressed = true;
					}
				});
				InputManager::Instance(window).doWhenKey(GLFW_KEY_R, false, [&]() {
					resetKeyPressed = false;
				});
				InputManager::Instance(window).doWhenKey(GLFW_KEY_T, true, [&]() {
					if (!resetKeyPressed) {
						camera.ResetPlane();
						AppendMessage("Camera Plane mode reset to default orientation and power");
						resetKeyPressed = true;
					}
				});
				InputManager::Instance(window).doWhenKey(GLFW_KEY_T, false, [&]() {
					resetKeyPressed = false;
				});
				InputManager::Instance(window).doWhenKey(GLFW_KEY_Y, true, [&]() {
					if (!resetKeyPressed)  {
						camera.ResetOrbit();
						AppendMessage("Camera Orbit mode reset to default orientation and distance");
						resetKeyPressed = true;
					}
				});
				InputManager::Instance(window).doWhenKey(GLFW_KEY_Y, false, [&]() {
					resetKeyPressed = false;
				});

				InputManager::Instance(window).doWhenKey(GLFW_KEY_ESCAPE, true, [&]() {
					Engine::Instance().Exit();
				});
			}
		},
		// Render
		[&]() {
			testScene.Render(
				// Pre-processing
				[]() {
				},
				// Rendering step
				[]() -> Image {


					return Image();
				},
				// Post-processing
				[]() {
				},
				// Fallback
				[&]() {
					AppendGraphicsError("Python Rendering failed, switching to opengl for now.");
					testScene.renderSettings->usePythonRendering = false;
				}
			);
		},
		// Post-processing
		[]() {},
		// ImGui Objects
		[&]() {
			{
				ImGui::Begin("Info Panel");
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
			}

            {
				ImGuiIO& io = ImGui::GetIO();
				ImGui::SetNextWindowBgAlpha(0.6f);
				ImGuiWindowFlags overlayFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize
											| ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
				ImGui::Begin("Overlay", nullptr, overlayFlags);
				ImVec2 mousePos = io.MousePos;
				ImGui::Text("Mouse: %.1f, %.1f (CapturedKB=%d, CapturedMouse=%d)", mousePos.x, mousePos.y, io.WantCaptureKeyboard ? 1 : 0, io.WantCaptureMouse ? 1 : 0);
				ImGui::End();
        	}

			{
				ImGui::Begin("Debugging Log Panel");
				// Persistent UI state
				static bool autoScroll = true;
				static bool pause = false;
				static bool showLow = true, showMedium = true, showHigh = true;
				static int maxEntries = 5000;

				// Controls
				ImGui::BeginChild("LogControls", ImVec2(0, 28), false);
				if (ImGui::Button("Clear")) {
					ClearDebugLog();
				}
				ImGui::SameLine();
				if (ImGui::Button("Copy")) {
					// Copy visible messages to clipboard
					std::string buf;
					for (size_t i = 0; i < debugLog.size(); ++i) {
						const auto& msg = debugLog[i];
						// apply severity filters
						if ((msg.severity == DebugMessage::LOW && !showLow) ||
							(msg.severity == DebugMessage::MEDIUM && !showMedium) ||
							(msg.severity == DebugMessage::HIGH && !showHigh))
							continue;
						buf += msg.ToString();
						buf += '\n';
					}
					ImGui::SetClipboardText(buf.c_str());
				}
				ImGui::SameLine();
				if (ImGui::Button("Pause")) {
					pause = !pause;
				}
				ImGui::SameLine();
				ImGui::Checkbox("Auto-scroll", &autoScroll);
				ImGui::SameLine();
				ImGui::PushItemWidth(100);
				ImGui::SliderInt("Max entries", &maxEntries, 100, 20000);
				ImGui::PopItemWidth();
				ImGui::EndChild();

				// Severity filters
				ImGui::BeginChild("LogFilters", ImVec2(0, 22), false);
				ImGui::Checkbox("Low", &showLow); ImGui::SameLine();
				ImGui::Checkbox("Medium", &showMedium); ImGui::SameLine();
				ImGui::Checkbox("High", &showHigh);
				ImGui::EndChild();

				// Trim log if it exceeds maxEntries
				if (debugLog.size() > static_cast<size_t>(maxEntries)) {
					debugLog.erase(debugLog.begin(), debugLog.begin() + (debugLog.size() - maxEntries));
				}

				// Log list (scrollable)
				ImGui::BeginChild("LogRegion", ImVec2(0, 300), true, ImGuiWindowFlags_HorizontalScrollbar);
				if (pause) {
					ImGui::TextDisabled("Paused");
				}

				for (size_t i = 0; i < debugLog.size(); ++i) {
					const auto& msg = debugLog[i];

					// Apply filters
					if (msg.severity == DebugMessage::LOW && !showLow) continue;
					if (msg.severity == DebugMessage::MEDIUM && !showMedium) continue;
					if (msg.severity == DebugMessage::HIGH && !showHigh) continue;

					// Choose color and short label
					ImVec4 col = ImVec4(1, 1, 1, 1);
					const char* sevLabel = "UNK";
					switch (msg.severity) {
					case DebugMessage::LOW:
						col = ImVec4(0.0f, 1.0f, 0.0f, 1.0f); sevLabel = "LOW"; break;
					case DebugMessage::MEDIUM:
						col = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); sevLabel = "MED"; break;
					case DebugMessage::HIGH:
						col = ImVec4(1.0f, 0.0f, 0.0f, 1.0f); sevLabel = "HIGH"; break;
					default:
						col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); sevLabel = "UNK"; break;
					}

					// Compose a short line: index message
					{
						const size_t MAX_LEN = 100;
						std::string text = msg.message;
						if (text.size() > MAX_LEN) {
							text = text.substr(0, MAX_LEN);
							text += "...";
						}
						std::string line = std::to_string(i) + ". " + text;

						// Selectable for copying/viewing details
						if (ImGui::Selectable(line.c_str())) {
							// on click, copy to clipboard
							ImGui::SetClipboardText(msg.ToString().c_str());
						}
					}

					ImGui::SameLine(ImGui::GetWindowWidth() + ImGui::GetScrollX() - 60);
					ImGui::TextColored(col, "%s", sevLabel);
				}

				// Auto-scroll to bottom if enabled
				if (autoScroll) {
					ImGui::SetScrollHereY(1.0f);
				}
				ImGui::EndChild();

				// Expanded view for selected message (optional)
				static int selectedIndex = -1;
				ImGui::BeginChild("DetailRegion", ImVec2(0, 100), true);
				if (ImGui::Button("Show Selected")) {
					if (selectedIndex >= 0 && selectedIndex < static_cast<int>(debugLog.size())) {
						// copy selected message to clipboard
						ImGui::SetClipboardText(debugLog[selectedIndex].ToString().c_str());
					}
				}
				ImGui::SameLine();
				ImGui::InputInt("Index", &selectedIndex);
				if (selectedIndex >= 0 && selectedIndex < static_cast<int>(debugLog.size())) {
					ImGui::Separator();
					ImGui::TextWrapped("%s", debugLog[selectedIndex].ToString().c_str());
				}
				ImGui::EndChild();
				ImGui::End();
			}
			
			{
				ImGui::Begin("Scene Panel");
				ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera.Position.x, camera.Position.y, camera.Position.z);
				ImGui::Text("Camera Orientation: (%.2f, %.2f, %.2f)", camera.Forward.x, camera.Forward.y, camera.Forward.z);
				ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", camera.Up.x, camera.Up.y, camera.Up.z);
				ImGui::Text("Camera Mode: %s", camera.mode == FIRST_PERSON ? "FIRST_PERSON" : camera.mode == PLANE ? "PLANE" : "ORBIT");
				ImGui::Checkbox("Use Python Rendering", &testScene.renderSettings->usePythonRendering);
				ImGui::Separator();
				ImGui::Text("Renderables in Scene:");
				for (size_t i = 0; i < testScene.renderables.size(); ++i) {
					std::string label = testScene.renderables[i]->GetName().empty() ? "Renderable " + std::to_string(i) : testScene.renderables[i]->GetName();
					if (ImGui::Selectable(label.c_str(), testScene.selectedRenderable == static_cast<int>(i))) {
						testScene.selectedRenderable = static_cast<int>(i);
					}
				}
				if (ImGui::Button("Print Selected Renderable Info")) {
					if (testScene.selectedRenderable >= 0 && testScene.selectedRenderable < static_cast<int>(testScene.renderables.size())) {
						AppendMessage("Selected Renderable: " +	testScene.renderables[testScene.selectedRenderable]->GetName());
					}
					else {
						AppendMessage("No renderable selected.");
					}
				}
				ImGui::End();
			}
			// TODO: Add hiearchy for objects 
		},
		[]() {
			AppendGraphicsError("Rendering failed, executing fallback function for engine update.");
		});

		
	// Clean up and exit
	litShaderProgram.Delete();
	materialProgram.Delete();
	unlitShaderProgram.Delete();
	
	floorMesh.CleanUp();
	lightMesh.CleanUp();
	testScene.CleanUp();
	
	Engine::Instance().CleanUp(window);
	Engine::Instance().Exit();

	return 0;
}
