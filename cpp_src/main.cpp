#include<filesystem>
namespace fs = std::filesystem;

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "shaderClass.h"
#include "textureClass.h"

// Create verticies for a triangle
GLfloat triVerts[] = {
	-0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f,        1.0f, 1.0f, 1.0f,      0.0f, 0.0f,// Lower left corner
	0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f,         1.0f, 1.0f, 1.0f,      2.0f, 0.0f, // Lower right corner
	0.0f, 0.5f * float(sqrt(3)) * 2 / 3, 0.0f,      1.0f, 1.0f, 1.0f,      1.0f, 2.0f, // Upper corner
	-0.5f / 2, 0.5f * float(sqrt(3)) / 6, 0.0f,     1.0f, 1.0f, 1.0f,     0.5f, 1.5f, // Inner left
	0.5f / 2, 0.5f * float(sqrt(3)) / 6, 0.0f,      1.0f, 1.0f, 1.0f,     1.5f, 1.5f, // Inner right
	0.0f, -0.5f * float(sqrt(3)) / 3, 0.0f,         1.0f, 1.0f, 1.0f,      1.0f, 0.5f, // Inner down
};

GLfloat sqrVerts[] = {
    -0.5f, -0.5f, 0.0f,     1.0f, 0.0f, 0.0f,       0.0f, 0.0f, // Bottom left corner
    -0.5f, 0.5f, 0.0f,      0.0f, 1.0f, 0.0f,       0.0f, 1.0f, // Top left corner
    0.5f, 0.5f, 0.0f,       0.0f, 0.0f, 1.0f,       1.0f, 1.0f, // Top right corner
    0.5f, -0.5f, 0.0f,      1.0f, 1.0f, 1.0f,       1.0f, 0.0f, // Bottom right corner
};

// Create indices for the triangle
GLuint triIndices[] = {
    0, 3, 5,
    3, 2, 4,
    5, 4, 1
};

GLuint sqrIndices[] = {
    0, 1, 2, // Upper left triangle
    0, 3, 2  // Lower right triangle
};

namespace fs = std::filesystem;

int main() {
    // Initialize GLFW
    if (!glfwInit()) return -1;

    const int width = 800;
    const int height = 800;

    // Define the vesion of GLFW in use
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // Define the OpenGL profile
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(width, height, "Sample Page", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Load OpenGL functions using GLAD
    gladLoadGL();

    // Set the viewport to the size of the window
    glViewport(0, 0, width, height);

    // Create a Shader Program from vertex and fragment shader files
    // Ensure the shader files are in the correct directory
    fs::path shaderDir = fs::current_path() / "shaders";
    if (!fs::exists(shaderDir)) {
        std::cerr << "Shader directory does not exist: " << shaderDir << std::endl;
        return -1;
    }
    const char* vertexShaderPath = "shaders/default.vert";
    const char* fragmentShaderPath = "shaders/default.frag";
    Shader shaderProgram(vertexShaderPath, fragmentShaderPath);

    // Generates Vertex Array Object and binds it
	VAO VAO1;
	VAO1.Bind();

	// Generates Vertex Buffer Object and links it to vertices
	VBO VBO1(sqrVerts, sizeof(sqrVerts));
	// Generates Element Buffer Object and links it to indices
	EBO EBO1(sqrIndices, sizeof(sqrIndices));


	// Links VBO to VAO
    // links position attribute (location = 0)
	VAO1.LinkAtrib(VBO1, 0, 3, GL_FLOAT, 8 * sizeof(GLfloat), (void*)0);
    // links color attribute (location = 1)
    VAO1.LinkAtrib(VBO1, 1, 3, GL_FLOAT, 8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
    // links texture coordinate attribute (location = 2)
    VAO1.LinkAtrib(VBO1, 2, 2, GL_FLOAT, 8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));


	// Unbind all buffers to prevent accidentally modifying them
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

    // Set the scale uniform location
    GLuint scaleID = glGetUniformLocation(shaderProgram.ID, "scale");

    const char* imagePath = "textures/pop_cat.jpg";
    Texture popCat(imagePath, GL_TEXTURE_2D, GL_TEXTURE0, GL_RGB, GL_UNSIGNED_BYTE);
    popCat.TexUnit(shaderProgram, "tex0", 0);

    while (!glfwWindowShouldClose(window)) {
        // Set a background color
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        // Clear the color buffer
        glClear(GL_COLOR_BUFFER_BIT);
        
        shaderProgram.Activate();

        glUniform1f(scaleID, 1.0f); // Set the scale uniform
        popCat.Bind(); // Bind the texture

        VAO1.Bind(); // Bind the VAO
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(window);

        glfwPollEvents();
    }

	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
    popCat.Delete();
    shaderProgram.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}