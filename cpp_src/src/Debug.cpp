#include "Debug.h"

std::string debugLog;

#include <sstream>

GLenum glCheckError_(const char* file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        AppendOpenGLError(error + " | " + std::string(file) + " (" + std::to_string(line) + ")");
    }
    return errorCode;
}

void APIENTRY glDebugOutput(GLenum source, 
                            GLenum type, 
                            unsigned int id, 
                            GLenum severity, 
                            GLsizei length, 
                            const char *message, 
                            const void *userParam)
{
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 

    std::string msgStr = message ? message : "";
    std::string out;
    out += "---------------\n";
    {
        std::ostringstream oss;
        oss << "Debug message (" << id << "): " << msgStr << "\n";
        out += oss.str();
    }

    std::string src;
    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             src = "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   src = "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: src = "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     src = "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     src = "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           src = "Source: Other"; break;
        default:                              src = "Source: Unknown"; break;
    }
    out += src + "\n";

    std::string typ;
    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               typ = "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typ = "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typ = "Type: Undefined Behaviour"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         typ = "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         typ = "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              typ = "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          typ = "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           typ = "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               typ = "Type: Other"; break;
        default:                                typ = "Type: Unknown"; break;
    }
    out += typ + "\n";

    std::string sev;
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         sev = "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       sev = "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          sev = "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: sev = "Severity: notification"; break;
        default:                             sev = "Severity: unknown"; break;
    }
    out += sev + "\n\n";

    // Choose appropriate append function based on severity/type
    if (severity == GL_DEBUG_SEVERITY_HIGH || type == GL_DEBUG_TYPE_ERROR) {
        AppendOpenGLError(out);
    } else if (severity == GL_DEBUG_SEVERITY_MEDIUM || severity == GL_DEBUG_SEVERITY_LOW) {
        AppendOpenGLWarning(out);
    } else {
        AppendOpenGLMessage(out);
    }
}

void EnableOpenGLDebugger()
{
    if (!glGetStringi) {
        AppendOpenGLWarning("glGetStringi not available! Ensure GLAD was initialized properly.");
        return;
    }

    // Check for KHR_debug extension
    bool hasKHRDebug = false;
    GLint numExtensions = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
    for (GLint i = 0; i < numExtensions; ++i) {
        const char* ext = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));
        if (ext && std::string(ext) == "GL_KHR_debug") {
            hasKHRDebug = true;
            break;
        }
    }
    if (!hasKHRDebug) {
        AppendOpenGLError("GL_KHR_debug extension not available!");
        return;
    }

    const GLubyte* versionStr = glGetString(GL_VERSION);
    if (versionStr) {
        AppendOpenGLMessage(std::string("OpenGL version string: ") + reinterpret_cast<const char*>(versionStr));
    } else {
        AppendOpenGLError("glGetString(GL_VERSION) returned NULL");
    }

    if (!glDebugMessageCallback) {
        AppendOpenGLError("glDebugMessageCallback not available (maybe GLAD not built with GL_KHR_debug).");
        return;
    }

    AppendOpenGLMessage("Enabling OpenGL debug output...");
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugOutput, nullptr);
}

void AppendMessage(const std::string& msg) {
    if (debugLog.size() + msg.size() > MAX_LOG_SIZE) {
        debugLog.clear(); // Clear log if exceeding max size
        debugLog += "[Debug Log Cleared Due to Size Limit]\n";
    }
    
    debugLog += "[Message] " + msg + "\n";
    std::cout << "[Message] " << msg << std::endl;
}
void AppendWarning(const std::string& warning) {
    if (debugLog.size() + warning.size() > MAX_LOG_SIZE) {
        debugLog.clear(); // Clear log if exceeding max size
        debugLog += "[Debug Log Cleared Due to Size Limit]\n";
    }

    debugLog += "[Warning] " + warning + "\n";
    std::cout << "[Warning] " << warning << std::endl;
}
void AppendError(const std::string& error) {
    if (debugLog.size() + error.size() > MAX_LOG_SIZE) {
        debugLog.clear(); // Clear log if exceeding max size
        debugLog += "[Debug Log Cleared Due to Size Limit]\n";
    }

    debugLog += "[Error] " + error + "\n";
    std::cerr << "[Error] " << error << std::endl;
}
void AppendOpenGLMessage(const std::string& msg) {
    if (debugLog.size() + msg.size() > MAX_LOG_SIZE) {
        debugLog.clear(); // Clear log if exceeding max size
        debugLog += "[Debug Log Cleared Due to Size Limit]\n";
    }

    debugLog += "[OpenGL Message] " + msg + "\n";
    std::cout << "[OpenGL Message] " << msg << std::endl;
}
void AppendOpenGLWarning(const std::string& warning) {
    if (debugLog.size() + warning.size() > MAX_LOG_SIZE) {
        debugLog.clear(); // Clear log if exceeding max size
        debugLog += "[Debug Log Cleared Due to Size Limit]\n";
    }

    debugLog += "[OpenGL Warning] " + warning + "\n";
    std::cout << "[OpenGL Warning] " << warning << std::endl;
}
void AppendOpenGLError(const std::string& error) {
    if (debugLog.size() + error.size() > MAX_LOG_SIZE) {
        debugLog.clear(); // Clear log if exceeding max size
        debugLog += "[Debug Log Cleared Due to Size Limit]\n";
    }

    debugLog += "[OpenGL Error] " + error + "\n";
    std::cerr << "[OpenGL Error] " << error << std::endl;
}
void PrintDebugLog() {
    if (debugLog.empty()) {
        std::cout << "[Debug Log is empty]" << std::endl;
        return;
    }
    for (const auto& line : debugLog) {
        std::cout << line;
    }
}
void ClearDebugLog() {
    debugLog.clear();
    std::cout << "[Debug Log cleared]" << std::endl;
}