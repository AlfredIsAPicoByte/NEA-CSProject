#version 460 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in vec3 VertColor;

uniform sampler2D u_albedoMap;
uniform bool u_hasAlbedo;

uniform float u_specularStrength;
uniform float u_ambient;

// Single light uniforms
// u_lightType: 0 = point, 1 = directional, 2 = spot
uniform int   u_lightType;
uniform vec3  u_lightPos;       
uniform vec3  u_lightDir;
uniform vec3  u_lightColor;
uniform vec2  u_lightRadius;    
uniform float u_lightIntensity;

// Camera / view
uniform vec3 u_viewPos; // world-space camera position

out vec4 FragColor;

vec3 calcLight(vec3 normal, vec3 viewDir, vec3 albedo) {
    vec3 lightDir;
    float attenuation = 1.0;

	if (u_lightType != 1) { // Point or spot light
		lightDir = normalize(u_lightPos - FragPos);
		float distance = length(u_lightPos - FragPos);
		attenuation = 1.0 / (1.0 + (distance * distance) / (u_lightRadius.x * u_lightRadius.y));
	} else { // Directional light
		lightDir = normalize(-u_lightDir);
	}

	// Diffuse
    vec3 diffuse = albedo * u_lightColor * max(dot(normal, lightDir), 0.0) * u_lightIntensity * attenuation;

    // Blinn-Phong specular
    vec3 H = normalize(lightDir + viewDir); // half-vector
    vec3 specular = u_lightColor * pow(max(dot(normal, H), 0.0), 32.0) * u_specularStrength * attenuation;

	if (u_lightType == 2) { // Spot light
		float theta = dot(lightDir, normalize(-u_lightDir));
		float epsilon = u_lightRadius.x - u_lightRadius.y;
		float intensity = clamp((theta - u_lightRadius.y) / epsilon, 0.0, 1.0);
		diffuse *= intensity;
		specular *= intensity;
	}

    // Ambient
    vec3 ambient = albedo * u_ambient;

    return ambient + diffuse + specular;
}

void main() {
    vec3 albedo = u_hasAlbedo ? texture(u_albedoMap, TexCoords).rgb : VertColor;
    vec3 N = normalize(Normal);

    // compute view direction (from fragment to view/camera)
    vec3 viewDir = normalize(u_viewPos - FragPos);

    vec3 color = calcLight(N, viewDir, albedo);

    // Apply simple gamma correction (sRGB)
    FragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
