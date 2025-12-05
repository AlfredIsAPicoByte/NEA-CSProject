#include "shaderClass.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

static bool checkShaderCompileStatus(GLuint shader, const std::string &name)
{
    if (shader == 0) {
        AppendError("Shader compile check called with shader == 0 (" + name + ")");
        return false;
    }

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success) return true;

    GLint logLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
    GLsizei bufSize = (logLen > 0) ? static_cast<GLsizei>(logLen) : 1;
    std::vector<char> logBuf(bufSize);
    GLsizei outLen = 0;
    glGetShaderInfoLog(shader, bufSize, &outLen, logBuf.data());
    std::string logStr(logBuf.data(), (outLen>0)?outLen:0);
    AppendError("Shader compile error (" + name + "):\n" + logStr);
    return false;
}

static bool checkProgramLinkStatus(GLuint program)
{
    if (program == 0) {
        AppendError("Program link check called with program == 0");
        return false;
    }

    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked) return true;

    GLint logLen = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
    GLsizei bufSize = (logLen > 0) ? static_cast<GLsizei>(logLen) : 1;
    std::vector<char> logBuf(bufSize);
    GLsizei outLen = 0;
    glGetProgramInfoLog(program, bufSize, &outLen, logBuf.data());
    std::string logStr(logBuf.data(), (outLen>0)?outLen:0);
    AppendError("Shader link error:\n" + logStr);
    return false;
}

// Ensure file reading returns content
std::string get_file_contents(const char* filename)
{
    namespace fs = std::filesystem;
    fs::path p(filename);

    // Try opening the path directly first
    if (fs::exists(p)) {
        std::ifstream in(p, std::ios::in | std::ios::binary);
        std::ostringstream contents;
        contents << in.rdbuf();
        return contents.str();
    }

    // Common candidate locations relative to current working directory
    std::vector<fs::path> candidates = {
        fs::current_path() / p,
        fs::current_path() / "cpp_src" / "src" / p,
        fs::current_path() / "cpp_src" / "src" / "shaders" / p.filename(),
        fs::current_path() / "shaders" / p.filename(),
        fs::current_path().parent_path() / "cpp_src" / "src" / "shaders" / p.filename(),
    };

    for (auto &c : candidates) {
        if (fs::exists(c)) {
            std::ifstream in(c, std::ios::in | std::ios::binary);
            std::ostringstream contents;
            contents << in.rdbuf();
            AppendMessage(std::string("Loaded shader from: ") + c.string());
            return contents.str();
        }
    }

    // Last resort: walk up parent directories and search under each root for the filename
    fs::path root = fs::current_path();
    int maxUp = 6;
    for (int depth = 0; depth < maxUp && !root.empty(); ++depth) {
        // try direct known shader folder
        fs::path candidate = root / "cpp_src" / "src" / "shaders" / p.filename();
        if (fs::exists(candidate)) {
            std::ifstream in(candidate, std::ios::in | std::ios::binary);
            std::ostringstream contents;
            contents << in.rdbuf();
            AppendMessage(std::string("Found shader in parent search: ") + candidate.string());
            return contents.str();
        }

        // recursive search under this root (safe since roots are limited)
        for (auto& entry : fs::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().filename() == p.filename()) {
                std::ifstream in(entry.path(), std::ios::in | std::ios::binary);
                std::ostringstream contents;
                contents << in.rdbuf();
                AppendMessage(std::string("Found shader by recursive parent search: ") + entry.path().string());
                return contents.str();
            }
        }

        // move up one level
        if (root.has_parent_path()) root = root.parent_path(); else break;
    }

    AppendError(std::string("Failed to open shader file: ") + filename);
    return std::string();
}

// Constructor that build the Shader Program from 2 different shaders
Shader::Shader(const char* vertexFileName, const char* fragmentFileName)
{
	// Path to the Vertex and Fragment shader files
	std::string vertexPath = std::string("shaders/") + vertexFileName;
	std::string fragmentPath = std::string("shaders/") + fragmentFileName;

	// Read vertexFile and fragmentFile and store the strings
	std::string vertexCode = get_file_contents(vertexPath.c_str());
	std::string fragmentCode = get_file_contents(fragmentPath.c_str());

	if (vertexCode.empty() || fragmentCode.empty()) {
        AppendError("One or more shader sources are empty; shader creation aborted.");
        ID = 0;
        return;
    }

    const char* vSrc = vertexCode.c_str();
    const char* fSrc = fragmentCode.c_str();

    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vSrc, NULL);
    glCompileShader(vertex);
    if (!checkShaderCompileStatus(vertex, vertexPath)) {
        glDeleteShader(vertex);
        ID = 0;
        return;
    }

    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fSrc, NULL);
    glCompileShader(fragment);
    if (!checkShaderCompileStatus(fragment, fragmentPath)) {
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        ID = 0;
        return;
    }

    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);

    // validate link
    if (!checkProgramLinkStatus(ID)) {
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        glDeleteProgram(ID);
        ID = 0;
        return;
    }

    // cleanup shader objects
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

// Activates the Shader Program
void Shader::Activate()
{
    if (ID == 0) {
        AppendError("Attempted to Activate a shader program with ID = 0");
        return;
    }
    GLint linked = 0;
    glGetProgramiv(ID, GL_LINK_STATUS, &linked);
    if (!linked) {
        AppendError("Attempted to Activate an unlinked shader program (ID=" + std::to_string(ID) + ")");
        return;
    }
    glUseProgram(ID);
}

// Deletes the Shader Program
void Shader::Delete()
{
	glDeleteProgram(ID);
}

GLint Shader::GetUniformLocation(const std::string& name) const
{
    if (name.empty()) {
        AppendError("Shader::GetUniformLocation called with empty name");
        return -1;
    }

    GLint linked = 0;
    glGetProgramiv(ID, GL_LINK_STATUS, &linked);
    if (!linked) {
		GLint logLen = 0;
		glGetProgramiv(ID, GL_INFO_LOG_LENGTH, &logLen);
		std::string log((logLen>0)?logLen:1, '\0');
		glGetProgramInfoLog(ID, logLen, nullptr, &log[0]);
		AppendError("Shader link error: " + log);
        return -1;
    }

    return glGetUniformLocation(ID, name.c_str());
}

// Replace each set* implementation to use GetUniformLocation and guard invalid loc
void Shader::setBool(const std::string &name, bool value) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform1i(loc, (int)value);
}
void Shader::setInt(const std::string &name, int value) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform1i(loc, value);
}
void Shader::setFloat(const std::string &name, float value) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform1f(loc, value);
}
void Shader::setVec2(const std::string &name, const glm::vec2 &value) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform2fv(loc, 1, &value[0]);
}
void Shader::setVec2(const std::string &name, float x, float y) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform2f(loc, x, y);
}
void Shader::setVec3(const std::string &name, const glm::vec3 &value) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform3fv(loc, 1, &value[0]);
}
void Shader::setVec3(const std::string &name, float x, float y, float z) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform3f(loc, x, y, z);
}
void Shader::setVec4(const std::string &name, const glm::vec4 &value) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform4fv(loc, 1, &value[0]);
}
void Shader::setVec4(const std::string &name, float x, float y, float z, float w) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniform4f(loc, x, y, z, w);
}
void Shader::setMat2(const std::string &name, const glm::mat2 &mat) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniformMatrix2fv(loc, 1, GL_FALSE, &mat[0][0]);
}
void Shader::setMat3(const std::string &name, const glm::mat3 &mat) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniformMatrix3fv(loc, 1, GL_FALSE, &mat[0][0]);
}
void Shader::setMat4(const std::string &name, const glm::mat4 &mat) const
{
    GLint loc = GetUniformLocation(name);
    if (loc == -1) return;
    glUniformMatrix4fv(loc, 1, GL_FALSE, &mat[0][0]);
}

// Checks if the different Shaders have compiled properly
void Shader::compileErrors(unsigned int shader, const char* type)
{
	GLint hasCompiled;
	char infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &hasCompiled);
		if (hasCompiled == GL_FALSE)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "SHADER_COMPILATION_ERROR for:" << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &hasCompiled);
		if (hasCompiled == GL_FALSE)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "SHADER_LINKING_ERROR for:" << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}
