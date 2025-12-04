#ifndef MESH_CLASS_H
#define MESH_CLASS_H

#include <string>
#include <vector>

#include "VAO.h"
#include "EBO.h"
#include "cameraClass.h"
#include "textureClass.h"
#include "IRenderable.h"
#include "Debugger.h"

class Mesh : public IRenderable
{
public:
	std::vector <Vertex> vertices;
	std::vector <GLuint> indices;
	std::vector <Texture> textures;

	glm::mat4 modelMatrix = glm::mat4(1.0f);
	// Store VAO in public so it can be used in the Draw function
	VAO VAO;

	// Initializes the mesh
	Mesh(std::vector <Vertex>& vertices, std::vector <GLuint>& indices, std::vector <Texture>& textures);

	// Draws the mesh using the provided model matrix
	void Draw(Shader& shader, Camera& camera) override;

	void CleanUp();

	const std::string& GetName() const {
		return name;
	}
	glm::mat4 GetModelMatrix() const {
		return modelMatrix;
	}
private:
	std::string name = "";
	int id;
};
#endif