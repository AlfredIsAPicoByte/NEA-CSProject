#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;

out vec3 color;
out vec2 texCoord;

uniform float scale;

void main()
{
   vec3 scaledPos = aPos + aPos * scale;
   gl_Position = vec4(scaledPos.x, scaledPos.y, scaledPos.z, 1.0);
   color = aColor;
   texCoord = aTexCoord;
}