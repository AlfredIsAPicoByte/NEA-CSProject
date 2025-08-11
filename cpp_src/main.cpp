#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "shaderClass.h"

// Create verticies for a triangle
GLfloat vertices[] = {
	-0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f,        1.0f, 1.0f, 1.0f,      0.0f, 0.0f,// Lower left corner
	0.5f, -0.5f * float(sqrt(3)) / 3, 0.0f,         1.0f, 1.0f, 1.0f,      2.0f, 0.0f, // Lower right corner
	0.0f, 0.5f * float(sqrt(3)) * 2 / 3, 0.0f,      1.0f, 1.0f, 1.0f,      1.0f, 2.0f, // Upper corner
	-0.5f / 2, 0.5f * float(sqrt(3)) / 6, 0.0f,     1.0f, 1.0f, 1.0f,     0.5f, 1.5f, // Inner left
	0.5f / 2, 0.5f * float(sqrt(3)) / 6, 0.0f,      1.0f, 1.0f, 1.0f,     1.5f, 1.5f, // Inner right
	0.0f, -0.5f * float(sqrt(3)) / 3, 0.0f,         1.0f, 1.0f, 1.0f,      1.0f, 0.5f, // Inner down
};

// Create indices for the triangle
GLuint indices[] = {
    0, 3, 5,
    3, 2, 4,
    5, 4, 1
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

    // Create a Shader Program
    // Ensure the shader files exist in the expected directory
    const char* vertexShaderPath = "shaders/default.vert";
    const char* fragmentShaderPath = "shaders/default.frag";
    Shader shaderProgram(vertexShaderPath, fragmentShaderPath);

    // Generates Vertex Array Object and binds it
	VAO VAO1;
	VAO1.Bind();

	// Generates Vertex Buffer Object and links it to vertices
	VBO VBO1(vertices, sizeof(vertices));
	// Generates Element Buffer Object and links it to indices
	EBO EBO1(indices, sizeof(indices));

	// Links VBO to VAO
	VAO1.LinkAtrib(VBO1, 0, 3, GL_FLOAT, 8 * sizeof(GLfloat), (void*)0);
    VAO1.LinkAtrib(VBO1, 1, 3, GL_FLOAT, 8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
    VAO1.LinkAtrib(VBO1, 2, 2, GL_FLOAT, 8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
	// Unbind all to prevent accidentally modifying them
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

    GLuint scaleID = glGetUniformLocation(shaderProgram.ID, "scale");

    int widthImg, heightImg, channels;
    unsigned char* data = stbi_load("textures/pop_cat.jpg", &widthImg, &heightImg, &channels, 0);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, widthImg, heightImg, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint tex0ID = glGetUniformLocation(shaderProgram.ID, "tex0");
    shaderProgram.Activate();
    glUniform1i(tex0ID, 0); // Set the texture unit to 0

    while (!glfwWindowShouldClose(window)) {
        // Set a background color
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        // Clear the color buffer
        glClear(GL_COLOR_BUFFER_BIT);
        
        shaderProgram.Activate();

        glUniform1f(scaleID, 0.5f); // Set the scale uniform
        glBindTexture(GL_TEXTURE_2D, textureID);
        
        VAO1.Bind();
        
        glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(window);

        glfwPollEvents();
    }

	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
    glDeleteTextures(1, &textureID);
    shaderProgram.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}