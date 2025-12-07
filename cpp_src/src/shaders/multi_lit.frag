#version 460 core

#define MAX_LIGHTS 64

layout(std140, binding = 2) uniform LightBlock {
    int    u_lightCount;
    // pad to 16 bytes
    vec3   _pad0;
    // arrays of vec4 (xyz = pos/dir, w = radius/type/hint)
    vec4   u_lightPosRadius[MAX_LIGHTS];      // xyz = position, w = radius (for point)
    vec4   u_lightColorIntensity[MAX_LIGHTS]; // rgb = color, w = intensity
    vec4   u_lightDirType[MAX_LIGHTS];        // xyz = direction (for dir/spot), w = type (0=point,1=dir,2=spot)
};

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 VertColor;

uniform float u_specularStrength;
uniform float u_ambient;
uniform sampler2D u_albedoMap;
uniform bool u_hasAlbedo;

out vec4 FragColor;

vec3 calcLight(int i, vec3 N, vec3 V, vec3 albedo) {
    vec3 Lpos = u_lightPosRadius[i].xyz;
    float radius = u_lightPosRadius[i].w;
    int type = int(u_lightDirType[i].w + 0.5);
    vec3 lightCol = u_lightColorIntensity[i].rgb * u_lightColorIntensity[i].w;
    vec3 L;
    float att = 1.0;
    if (type == 1) { // directional
        L = normalize(u_lightDirType[i].xyz);
    } else { // point
        L = normalize(Lpos - FragPos);
        float dist = length(Lpos - FragPos);
        att = clamp(1.0 - dist / max(radius, 0.0001), 0.0, 1.0);
    }
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * lightCol * NdotL * att + albedo * u_ambient;
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    vec3 specular = lightCol * spec * att * u_specularStrength;
    return diffuse + specular;
}

void main() {
    vec3 albedo = u_hasAlbedo ? texture(u_albedoMap, TexCoord).rgb : VertColor;
    vec3 N = normalize(Normal);
    vec3 V = normalize(-FragPos); // replace with viewPos if available

    vec3 color = vec3(0.0);
    for (int i = 0; i < u_lightCount; ++i) {
        color += calcLight(i, N, V, albedo);
    }
    FragColor = vec4(pow(color, vec3(1.0/2.2)), 1.0);
}