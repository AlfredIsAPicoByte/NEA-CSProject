#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aTangent; // xyz = tangent, w = handedness

// Imports the camera matrix from the main function
uniform mat4 u_camMatrix;
uniform vec3 u_viewDir;
// Imports the model matrix from the main function
uniform mat4 u_model;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 VertColor;
out vec3 Tangent;
out vec3 Bitangent;
out vec3 ViewDir;

void main()
{
    vec4 worldPos4 = u_model * vec4(aPosition, 1.0);
    FragPos = worldPos4.xyz;

    // Normal matrix from model (handles non-uniform scale)
    mat3 normalMatrix = transpose(inverse(mat3(u_model)));
    Normal = normalize(normalMatrix * aNormal);

    // Tangent space (uses tangent.w as handedness)
    vec3 T = normalize(normalMatrix * aTangent.xyz);
    float handedness = aTangent.w;
    vec3 B = cross(Normal, T) * handedness;

    Tangent = T;
    Bitangent = B;

    TexCoord = aTexCoord;
    ViewDir = u_viewDir;

    gl_Position = u_camMatrix * worldPos4;
}