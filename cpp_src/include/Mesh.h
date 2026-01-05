#pragma once

#include <string>
#include <vector>

#include "VAO.h"
#include "EBO.h"
#include "textureClass.h"
#include "IRenderable.hpp"
#include "Debugger.h"

class Mesh : public IRenderable
{
public:
    std::vector <Vertex> vertices;
    std::vector <GLuint> indices;
    std::vector <Texture> textures;

    // Store VAO in public so it can be used in the Draw function
    VAO VAO;

    // Default constructor
    Mesh() = default;

    // Standard constructor. Creates all the buffer objects/arrays
    Mesh(std::vector <Vertex>& vertices, std::vector <GLuint>& indices, std::vector <Texture>& textures);

    // Draws the mesh
    void Draw(Shader& shader, Camera& camera) override;

    void CleanUp() override;
protected:
	void SerializeFields(json& j) const override;
    void DeserializeFields(const json& j) override;
private:
    // Helper to generate VBO/EBO/VAO. 
    // Used by both the Constructor and DeserializeFields.
    void setupMesh();
};