#version 330 core
// Designed for a sword model. Expects tangents/bitangents provided by the vertex shader.

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in vec3 Tangent;
in vec3 Bitangent;

out vec4 FragColor;

// Material textures
uniform sampler2D albedoMap;              // RGB albedo (sRGB)
uniform sampler2D normalMap;              // normal in tangent space (RGB)
uniform sampler2D metallicRoughnessMap;   // R = metallic, G = roughness (linear)
uniform sampler2D aoMap;                  // ambient occlusion (linear)
uniform sampler2D emissiveMap;            // optional emissive (sRGB)

// Camera
uniform vec3 camPos;

// Lights (support up to 4)
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];
uniform int lightCount;

// constants
const float PI = 3.14159265359;

// --- helpers ---
vec3 SRGBToLinear(vec3 c) {
    return pow(c, vec3(2.2));
}
vec3 LinearToSRGB(vec3 c) {
    return pow(c, vec3(1.0/2.2));
}

// get normal from normal map, reconstruct in world space using TBN
vec3 getNormalFromMap()
{
    vec3 tangentNormal = texture(normalMap, TexCoords).rgb;
    tangentNormal = tangentNormal * 2.0 - 1.0;

    vec3 T = normalize(Tangent);
    vec3 B = normalize(Bitangent);
    vec3 N = normalize(Normal);
    mat3 TBN = mat3(T, B, N);
    return normalize(TBN * tangentNormal);
}

// PBR functions (GGX / Schlick)
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / max(denom, 1e-5);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    float denom = NdotV * (1.0 - k) + k;
    return NdotV / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// --- main ---
void main()
{
    // Material fetch
    vec3 albedo = SRGBToLinear(texture(albedoMap, TexCoords).rgb);
    vec3 emissive = SRGBToLinear(texture(emissiveMap, TexCoords).rgb);
    float ao = texture(aoMap, TexCoords).r;
    vec2 mr = texture(metallicRoughnessMap, TexCoords).rg;
    float metallic = clamp(mr.r, 0.0, 1.0);
    float roughness = clamp(mr.g, 0.04, 1.0); // min roughness to avoid singularities

    // Normal (world)
    vec3 N = getNormalFromMap();
    vec3 V = normalize(camPos - FragPos);

    // Calculate reflectance at normal incidence; metals use albedo as F0
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // Accumulate lighting
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < lightCount; ++i)
    {
        vec3 L = normalize(lightPositions[i] - FragPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - FragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator    = NDF * G * F;
        float denom       = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 1e-5;
        vec3 specular     = numerator / denom;

        // kS is energy for specular, kD diffuse component
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Ambient (approximate using AO and a small IBLa)
    vec3 ambient = vec3(0.03) * albedo * ao;

    vec3 color = ambient + Lo + emissive;

    // tone mapping (ACES approximation) and gamma
    // simple reinhard then gamma for brevity
    color = color / (color + vec3(1.0));
    color = LinearToSRGB(color);

    FragColor = vec4(color, 1.0);
}