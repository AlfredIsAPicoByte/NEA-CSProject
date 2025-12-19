#include "Mesh.h"

Mesh::Mesh(std::vector <Vertex>& vertices, std::vector <GLuint>& indices, std::vector <Texture>& textures)
{
    Mesh::vertices = vertices;
    Mesh::indices = indices;
    Mesh::textures = textures;

    VAO.Bind();
    // Generates Vertex Buffer Object and links it to vertices
    VBO VBO(vertices);
    // Generates Element Buffer Object and links it to indices
    EBO EBO(indices);
    // Links VBO attributes such as coordinates and colors to VAO
    VAO.LinkAttrib(VBO, 0, 3, GL_FLOAT, sizeof(Vertex), (void*)0); // position
    VAO.LinkAttrib(VBO, 1, 3, GL_FLOAT, sizeof(Vertex), (void*)(3 * sizeof(float))); // normal
    VAO.LinkAttrib(VBO, 2, 3, GL_FLOAT, sizeof(Vertex), (void*)(6 * sizeof(float))); // color
    VAO.LinkAttrib(VBO, 3, 2, GL_FLOAT, sizeof(Vertex), (void*)(9 * sizeof(float))); // texUV
    VAO.LinkAttrib(VBO, 4, 4, GL_FLOAT, sizeof(Vertex), (void*)(11 * sizeof(float))); // tangent

    // Unbind all to prevent accidentally modifying them
    VAO.Unbind();
}

void Mesh::Draw(Shader& shader, Camera& camera)
{
    // Bind shader to be able to access uniforms
    shader.Activate();
    VAO.Bind();

    // Keep track of how many of each type of textures we have
    unsigned int numDiffuse = 0;
    unsigned int numSpecular = 0;
    unsigned int numNormal = 0;

    // Remember the texture unit of the first diffuse texture (if any)
    int firstDiffuseUnit = -1;

    for (unsigned int i = 0; i < textures.size(); i++)
    {
        std::string num;
        std::string type = textures[i].type;
        if (type == "diffuse") {
            num = std::to_string(numDiffuse++);
            // record the first diffuse unit
            if (firstDiffuseUnit == -1) firstDiffuseUnit = static_cast<int>(textures[i].unit);
        }
        else if (type == "specular")
            num = std::to_string(numSpecular++);
        else if (type == "normal")
            num = std::to_string(numNormal++);
        else
            num = std::to_string(i);

        // Ensure the shader sampler uniform is set to the actual unit used by this Texture object.
        textures[i].texUnit(shader, (type + num).c_str(), textures[i].unit);
        textures[i].Bind();
    }

    // If we found at least one diffuse texture, tell the shader which sampler to use.
    if (firstDiffuseUnit >= 0) {
        shader.setInt("u_albedoMap", firstDiffuseUnit);
        shader.setBool("u_hasAlbedo", true);
    } else {
        shader.setBool("u_hasAlbedo", false);
    }

    // Take care of the camera Matrix: upload camera position (if needed) and cam/model matrices
    // Some shaders expect u_viewPos (fragment shader) while others might use camPos; we set u_viewPos from main as well.
	shader.setVec3("u_viewPos", camera.Position);
	shader.setVec3("u_viewDir", camera.Forward);
    camera.SetModelMatrixUniform(shader, "u_camMatrix");

    // Push the model matrix to the vertex shader (must match the vertex shader's uniform name)
	shader.setMat4("u_model", GetModelMatrix());

    // Draw the actual mesh
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
}

void Mesh::CleanUp()
{
    VAO.Delete();
}

json Mesh::ToJSON() const
{
    json j;
    j["vertices"] = json::array();
    for (const auto& v : vertices) {
        json jv;
        jv["position"] = { v.Position.x, v.Position.y, v.Position.z };
        jv["normal"] = { v.Normal.x, v.Normal.y, v.Normal.z };
        jv["color"] = { v.color.x, v.color.y, v.color.z };
        jv["uv"] = { v.TexCoords.x, v.TexCoords.y };
        jv["tangent"] = { v.Tangent.x, v.Tangent.y, v.Tangent.z, v.Tangent.w };
        j["vertices"].push_back(jv);
    }
    j["indices"] = indices;
    j["textures"] = json::array();
    for (const auto& t : textures) {
        json jt;
        jt["ID"] = t.ID;
        jt["type"] = t.type;
        jt["unit"] = t.unit;
        j["textures"].push_back(jt);
    }
    j["constructor"] = "Mesh";
    return j;
}

void Mesh::FromJSON(const json& j)
{
    vertices.clear();
    for (const auto& jv : j["vertices"]) {
        Vertex v;
        v.Position = glm::vec3(jv["position"][0], jv["position"][1], jv["position"][2]);
        v.Normal = glm::vec3(jv["normal"][0], jv["normal"][1], jv["normal"][2]);
        v.color = glm::vec3(jv["color"][0], jv["color"][1], jv["color"][2]);
        v.TexCoords = glm::vec2(jv["uv"][0], jv["uv"][1]);
        v.Tangent = glm::vec4(jv["tangent"][0], jv["tangent"][1], jv["tangent"][2], jv["tangent"][3]);
        vertices.push_back(v);
    }
    indices = j["indices"].get<std::vector<GLuint>>();
    textures.clear();
    for (const auto& jt : j["textures"]) {
        Texture t("", "", 0); // dummy initialization
        t.ID = jt["ID"];
        t.type = jt["type"].get<std::string>().c_str();
        t.unit = jt["unit"];
        textures.push_back(t);
    }
}