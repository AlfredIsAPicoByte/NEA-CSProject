#include "Mesh.h"

Mesh::Mesh(std::vector <Vertex>& vertices, std::vector <GLuint>& indices, std::vector <Texture>& textures)
{
	Mesh::vertices = vertices;
	Mesh::indices = indices;
	Mesh::textures = textures;

	VAO.Bind();
	VBO vbo(vertices);
	EBO ebo(indices);

	const GLsizei stride = static_cast<GLsizei>(sizeof(Vertex));
	VAO.LinkAttrib(vbo, 0, 3, GL_FLOAT, stride, (void*)0);                             // aPos
	VAO.LinkAttrib(vbo, 1, 3, GL_FLOAT, stride, (void*)(3 * sizeof(float)));          // aNormal
	VAO.LinkAttrib(vbo, 2, 3, GL_FLOAT, stride, (void*)(6 * sizeof(float)));          // aColor (if used)
	VAO.LinkAttrib(vbo, 3, 2, GL_FLOAT, stride, (void*)(9 * sizeof(float)));          // aTexCoord
	VAO.LinkAttrib(vbo, 4, 4, GL_FLOAT, stride, (void*)(11 * sizeof(float)));         // aTangent
	VAO.Unbind();
}

void Mesh::Draw(Shader &shader, Camera &camera)
{
    // Activate shader and set common uniforms
    shader.Activate(); // safe: Activate should check program linked
    shader.setMat4("view", camera.viewMatrix);
    shader.setMat4("projection", camera.projectionMatrix);

    // model & normal matrix
    shader.setMat4("model", modelMatrix);
    glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(modelMatrix)));
    shader.setMat4("normalMatrix", glm::mat4(normalMat)); // or setMat3 if shader expects mat3

    // Bind textures: use texture slot = index in textures vector (or stored slot)
    bool hasAlbedo = false;
    for (size_t t = 0; t < textures.size(); ++t) {
        const Texture &tex = textures[t];
        if (tex.ID == 0) continue;
        GLuint unit = static_cast<GLuint>(t); // texture unit
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, tex.ID);

        if (tex.type == "diffuse" || tex.type == "albedo") {
            shader.setInt("u_albedoMap", (int)unit);
            hasAlbedo = true;
        } else if (tex.type == "normal") {
            shader.setInt("u_normalMap", (int)unit);
        } else if (tex.type == "metallicRoughness" || tex.type == "specular") {
            shader.setInt("u_metallicRoughnessMap", (int)unit);
        }
    }
    shader.setBool("u_hasAlbedo", hasAlbedo ? 1 : 0);

    // Draw
    VAO.Bind();
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
    VAO.Unbind();

    // restore default
    glActiveTexture(GL_TEXTURE0);
}

void Mesh::CleanUp()
{
	VAO.Delete();
}