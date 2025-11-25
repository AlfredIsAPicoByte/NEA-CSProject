#ifndef VBO_CLASS_H
#define VBO_CLASS_H

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <vector>

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 texUV;
	glm::vec3 tangent;

	void calculateNormal(const glm::vec3& edge1, const glm::vec3& edge2)
	{
		normal = glm::normalize(glm::cross(edge1, edge2));
	}
	void calculateTangent(const glm::vec3& edge1, const glm::vec3& edge2, const glm::vec2& deltaUV1, const glm::vec2& deltaUV2)
	{
		float f = 1.0f;
		float denom = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
		if (fabs(denom) > 1e-6f) f = 1.0f / denom;

		tangent = f * (edge1 * deltaUV2.y - edge2 * deltaUV1.y);
	}
};

class VBO
{
public:
	// Reference ID of the Vertex Buffer Object
	GLuint ID;
	// Constructor that generates a Vertex Buffer Object and links it to vertices
	VBO(std::vector<Vertex>& vertices);

	// Binds the VBO
	void Bind();
	// Unbinds the VBO
	void Unbind();
	// Deletes the VBO
	void Delete();
};

#endif