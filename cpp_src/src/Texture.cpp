#include "Texture.h"

Texture::Texture(const char* image, const char* texType, GLuint slot, GLenum pixelType)
{
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;

    // direct path provided
    candidates.push_back(fs::path(image));

    // if file is relative like "planks.png", try textures/ and also cpp_src/src/textures
    candidates.push_back(fs::path("textures") / image);
    candidates.push_back(fs::path("cpp_src/src/textures") / image);
    // also try the exe directory + textures folder
    candidates.push_back(fs::current_path() / "textures" / image);

    fs::path found;
    for (auto &c : candidates) {
        if (fs::exists(c)) { found = fs::canonical(c); break; }
    }

    if (found.empty()) {
        // last-resort: search recursively under known resource root (expensive but helpful for debugging)
        fs::path searchRoot = fs::current_path();
        for (auto& entry : fs::recursive_directory_iterator(searchRoot)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().filename() == fs::path(image).filename()) { found = entry.path(); break; }
        }
    }

    std::cerr << "Texture lookup for '" << image << "': ";
    if (!found.empty()) std::cerr << found.string() << std::endl;
    else {
        std::cerr << "NOT FOUND (tried candidates and recursive search)" << std::endl;
        throw std::runtime_error(std::string("Texture source is null for file: ") + image);
    }

    // then call stbi_load(found.string().c_str(), ...)
    // Assigns the type of the texture ot the texture object
    type = texType;
    unit = slot;

    glGenTextures(1, &ID);
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, ID);

    // Stores the width, height, and the number of color channels of the image
    int widthImg, heightImg, numColCh;
    // Flips the image so it appears right side up
    // glTF images should NOT be flipped when loaded
    stbi_set_flip_vertically_on_load(false);
    // Reads the image from the resolved file path
    std::string imagePath = found.string();
    unsigned char* bytes = stbi_load(imagePath.c_str(), &widthImg, &heightImg, &numColCh, 0);
    if (!bytes) {
        std::string msg = std::string("Texture source is null for file: ") + imagePath;
        AppendError(msg);
        throw std::runtime_error(msg);
    }

    GLenum externalFormat = GL_RGB;
    GLenum internalFormat = GL_RGB8;
    if (numColCh == 1) {
        externalFormat = GL_RED;
        internalFormat = GL_R8;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // important for 1-byte rows
    } else if (numColCh == 3) {
        externalFormat = GL_RGB;
        internalFormat = GL_RGB8;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    } else if (numColCh == 4) {
        externalFormat = GL_RGBA;
        internalFormat = GL_RGBA8;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    } else {
        stbi_image_free(bytes);
        throw std::runtime_error("Unsupported number of channels in texture: " + std::to_string(numColCh));
    }

    AppendMessage("Loaded texture: " + imagePath + " (Width: " + std::to_string(widthImg) + ", Height: " + std::to_string(heightImg) + ", Channels: " + std::to_string(numColCh) + ")");

    // Configures the type of algorithm that is used to make the image smaller or bigger
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Configures the way the texture repeats (if it does at all)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Extra lines in case you choose to use GL_CLAMP_TO_BORDER
    // float flatColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    // glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, flatColor);

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, widthImg, heightImg, 0, externalFormat, pixelType, bytes);
    glGenerateMipmap(GL_TEXTURE_2D);
    // reset alignment if you changed it elsewhere
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    // Deletes the image data as it is already in the OpenGL Texture object
    stbi_image_free(bytes);

    // Unbinds the OpenGL Texture object so that it can't accidentally be modified
    glBindTexture(GL_TEXTURE_2D, 0);

    AppendMessage("Texture created with ID: " + std::to_string(ID));
}

void Texture::texUnit(Shader& shader, const char* uniform, GLuint unit)
{
	// Gets the location of the uniform
	GLuint texUni = glGetUniformLocation(shader.ID, uniform);
	// Shader needs to be activated before changing the value of a uniform
	shader.Activate();
	// Sets the value of the uniform
	glUniform1i(texUni, unit);
}

void Texture::Bind()
{
	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, ID);
}

void Texture::Unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::Delete()
{
	glDeleteTextures(1, &ID);
}