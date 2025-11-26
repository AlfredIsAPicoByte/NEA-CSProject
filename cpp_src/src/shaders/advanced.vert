#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aTangent; // xyz = tangent, w = handedness

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uCameraPos; // world-space camera position

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vTangent;
out vec3 vBitangent;
out vec3 vViewDir;

void main()
{
    vec4 worldPos4 = uModel * vec4(aPosition, 1.0);
    vWorldPos = worldPos4.xyz;

    // Normal matrix from model (handles non-uniform scale)
    mat3 normalMatrix = transpose(inverse(mat3(uModel)));
    vNormal = normalize(normalMatrix * aNormal);

    // Tangent space (uses tangent.w as handedness)
    vec3 T = normalize(normalMatrix * aTangent.xyz);
    float handedness = aTangent.w;
    vec3 B = cross(vNormal, T) * handedness;

    vTangent = T;
    vBitangent = B;

    vTexCoord = aTexCoord;
    vViewDir = normalize(uCameraPos - vWorldPos);

    gl_Position = uProjection * uView * worldPos4;
}