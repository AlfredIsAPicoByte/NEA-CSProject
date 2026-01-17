#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;
layout (location = 3) in vec2 aTexCoords;
layout (location = 4) in vec4 aTangent;

// Outputs for the fragment shader (names/types must match)
// Outputs for the fragment shader (names/types must match)
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;
out vec3 VertColor;

// Imports the camera matrix from the main function
uniform mat4 u_camMatrix;
// Imports the model matrix from the main function
uniform mat4 u_model;

void main()
{
	// Compute world-space position
	vec4 worldPos = u_model * vec4(aPos, 1.0);
	FragPos = worldPos.xyz;

	// Transform normal to world space
	Normal = mat3(transpose(inverse(u_model))) * aNormal;

	// Pass through texture coordinates
	TexCoords = aTexCoords;

	// Pass through vertex color
	VertColor = aColor;
	// Final clip-space position
	gl_Position = u_camMatrix * worldPos;
}