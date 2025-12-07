# pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <json.hpp>

#include "Mesh.h"

using json = nlohmann::json;

class Model {
public:
	// Loads in a model from a file and stores tha information in 'data', 'JSON', and 'file'
	Model(const char* file);
	Model(Model&& other) noexcept;

	void Draw(Shader& shader, Camera& camera);

	void CleanUp();

	std::vector<Mesh>& GetMeshes();
	void SetName(const std::string& modelName);
	const std::string& GetName() const;
	glm::mat4 GetModelMatrixForMesh(unsigned int meshIndex) const;
	std::vector<glm::mat4> GetModelMatricesForAllMeshes() const;
	void SetModelMatrixForMesh(unsigned int meshIndex, const glm::mat4& modelMatrix);
	void SetModelMatricesForAllMeshes(const std::vector<glm::mat4>& modelMatrices);
private:
	std::string name;
	int id;
	
	// Variables for easy access
	const char* file;
	std::vector<unsigned char> data;
	json JSON;
	std::filesystem::path modelPath;

	// All the meshes and model matrices in the model
	std::vector<Mesh> meshes;
	std::vector<glm::mat4> matricesMeshes;

	// Prevents textures from being loaded twice
	std::vector<std::string> loadedTexName;
	std::vector<Texture> loadedTex;

	// Loads a single mesh by its index
	void loadMesh(unsigned int indMesh);

	// Traverses a node recursively, so it essentially traverses all connected nodes
	void traverseNode(unsigned int nextNode, glm::mat4 matrix = glm::mat4(1.0f));

	// Gets the binary data from a file
	std::vector<unsigned char> getData();
	// Interprets the binary data into floats, indices, and textures
	std::vector<float> getFloats(json accessor);
	std::vector<GLuint> getIndices(json accessor);
	std::vector<Texture> getTextures();

	// Assembles all the floats into vertices
	std::vector<Vertex> assembleVertices
	(
		std::vector<glm::vec3> positions, 
		std::vector<glm::vec3> normals, 
		std::vector<glm::vec2> texUVs
	);

	// Helps with the assembly from above by grouping floats
	std::vector<glm::vec2> groupFloatsVec2(std::vector<float> floatVec);
	std::vector<glm::vec3> groupFloatsVec3(std::vector<float> floatVec);
	std::vector<glm::vec4> groupFloatsVec4(std::vector<float> floatVec);

	// Computes the tangents for normal mapping
	void computeTangents(std::vector<Vertex>& verts, const std::vector<GLuint>& indices);
};