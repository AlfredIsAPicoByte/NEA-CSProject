#pragma once

#include <string>
#include <vector>

#include "VAO.h"
#include "EBO.h"
#include "textureClass.h"
#include "IRenderable.h"
#include "Debugger.h"

class Mesh : public IRenderable
{
public:
	std::vector <Vertex> vertices;
	std::vector <GLuint> indices;
	std::vector <Texture> textures;

	// Store VAO in public so it can be used in the Draw function
	VAO VAO;

	// Initializes the mesh
	Mesh(std::vector <Vertex>& vertices, std::vector <GLuint>& indices, std::vector <Texture>& textures);

	// Draws the mesh
	void Draw(Shader& shader, Camera& camera) override;

	void CleanUp() override;
	json ToJSON() const override;
	void FromJSON(const json& j) override;
};