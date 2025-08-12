#include "textureClass.h"

Texture::Texture(const char* imagePath, GLenum texType, GLenum slot, GLenum format, GLenum pixelType): type(texType) {
    // Load image
    int widthImg, heightImg, channels;
    stbi_set_flip_vertically_on_load(true); // Flip the image vertically
    unsigned char* imgBytes = stbi_load(imagePath, &widthImg, &heightImg, &channels, 0);

    glGenTextures(1, &ID);
    glActiveTexture(slot);
    glBindTexture(texType, ID);

    // Set texture parameters
    glTexParameteri(texType, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(texType, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(texType, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(texType, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexImage2D(texType, 0, GL_RGB, widthImg, heightImg, 0, format, pixelType, imgBytes);
    glGenerateMipmap(texType);

    stbi_image_free(imgBytes);
    glBindTexture(texType, 0);
};

void Texture::TexUnit(Shader& shader, const char* uniform, GLuint unit) {
    GLuint texUnit = glGetUniformLocation(shader.ID, uniform);
    shader.Activate();
    glUniform1i(texUnit, unit);
}

void Texture::Bind() {
    glActiveTexture(type);
    glBindTexture(type, ID);
}

void Texture::Unbind() {
    glBindTexture(type, 0);
}

void Texture::Delete() {
    glDeleteTextures(1, &ID);
}