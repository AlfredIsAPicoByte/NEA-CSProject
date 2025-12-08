#include "Debugger.h"

std::vector<DebugMessage> debugLog;

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
        AppendGraphicsError(error + " | " + std::string(file) + " (" + std::to_string(line) + ")");
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

    DebugMessage::Source src;
    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             src = DebugMessage::API; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   src = DebugMessage::WIN_SYS; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: src = DebugMessage::SHADER; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     src = DebugMessage::THIRD_PARTY; break;
        case GL_DEBUG_SOURCE_APPLICATION:     src = DebugMessage::APPLICATION; break;
        case GL_DEBUG_SOURCE_OTHER:           src = DebugMessage::OTHER_SRC; break;
        default:                              src = DebugMessage::UNKNOWN_SRC; break;
    }

    DebugMessage::Type typ;
    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               typ = DebugMessage::ERROR; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typ = DebugMessage::DEPRECATED_BEHAVIOR; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typ = DebugMessage::UNDEFINED_BEHAVIOR; break;
        case GL_DEBUG_TYPE_PORTABILITY:         typ = DebugMessage::PORTABILITY; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         typ = DebugMessage::PERFORMANCE; break;
        case GL_DEBUG_TYPE_MARKER:              typ = DebugMessage::MARKER; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          typ = DebugMessage::PUSH_GROUP; break;
        case GL_DEBUG_TYPE_POP_GROUP:           typ = DebugMessage::POP_GROUP; break;
        case GL_DEBUG_TYPE_OTHER:               typ = DebugMessage::OTHER_TYP; break;
        default:                                typ = DebugMessage::UNKNOWN_TYP; break;
    }

    DebugMessage::Severity sev;
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         sev = DebugMessage::HIGH; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       sev = DebugMessage::MEDIUM; break;
        case GL_DEBUG_SEVERITY_LOW:          sev = DebugMessage::LOW; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: sev = DebugMessage::NOTIFICATION; break;
        default:                             sev = DebugMessage::UNKNOWN_SEV; break;
    }
    
    DebugMessage msg(message, src, typ, sev);

    AppendDebugMessage(msg, true);
}

void EnableOpenGLDebugger()
{
    if (!glGetStringi) {
        AppendGraphicsWarning("glGetStringi not available! Ensure GLAD was initialized properly.");
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
        AppendGraphicsError("GL_KHR_debug extension not available!");
        return;
    }

    const GLubyte* versionStr = glGetString(GL_VERSION);
    if (versionStr) {
        AppendGraphicsMessage(std::string("OpenGL version string: ") + reinterpret_cast<const char*>(versionStr));
    } else {
        AppendGraphicsError("glGetString(GL_VERSION) returned NULL");
    }

    if (!glDebugMessageCallback) {
        AppendGraphicsError("glDebugMessageCallback not available (maybe GLAD not built with GL_KHR_debug).");
        return;
    }

    AppendGraphicsMessage("Enabling OpenGL debug output...");
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugOutput, nullptr);
}

// helper: compute approximate total size of the current debug log
static size_t getTotalLogSize()
{
    size_t total = 0;
    for (const auto &e : debugLog) {
        total += static_cast<size_t>(e.getApproxSize());
    }
    return total;
}

void AppendDebugMessage(const DebugMessage& msg, bool saveWhenFull) {
    // If the last entry is identical (same source/type/severity/message) -> increment its count
    if (!debugLog.empty()) {
        DebugMessage &last = debugLog.back();
        if (last.source == msg.source &&
            last.type == msg.type &&
            last.severity == msg.severity &&
            last.message == msg.message)
        {
            // estimate size after increment: conservative check using msg approx size
            if (willEntryExceedMaxLogSize(msg.getApproxSize())) {
                Time time;
                time.update();
                if (saveWhenFull) SaveDebugLogToFile("debug_log_" + std::to_string(time.lastFrame.time_since_epoch().count()) + ".txt");
                ClearDebugLog();
                AppendMessage("Debug Log Cleared Due to Size Limit");
                if (saveWhenFull) AppendMessage("Debug Log Saved to debug_log_" + std::to_string(time.lastFrame.time_since_epoch().count()) + ".txt before clearing.");
            }

            last.count += 1;
            return;
        }
    }

    // non-duplicate entry: check size and rotate if needed
    if (willEntryExceedMaxLogSize(msg.getApproxSize())) {
        Time time;
        time.update(); // Ensure time is updated for timestamping if needed
        if (saveWhenFull) SaveDebugLogToFile("debug_log_" + std::to_string(time.lastFrame.time_since_epoch().count()) + ".txt");

        ClearDebugLog(false); // Clear log if exceeding max size
        AppendMessage("Debug Log Cleared Due to Size Limit");

        if (saveWhenFull) AppendMessage("Debug Log Saved to debug_log_" + std::to_string(time.lastFrame.time_since_epoch().count()) + ".txt before clearing.");
    }
    debugLog.push_back(msg);
}
void AppendDebugMessage(const std::string& msg, DebugMessage::Source source, DebugMessage::Type type, DebugMessage::Severity severity) {
    DebugMessage debugMsg(msg, source, type, severity);
    AppendDebugMessage(debugMsg);
}

void AppendMessage(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::APPLICATION, DebugMessage::Type::MESSAGE, severity);
}
void AppendWarning(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::APPLICATION, DebugMessage::Type::WARNING, severity);
}
void AppendError(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::APPLICATION, DebugMessage::Type::ERROR, severity);
}
void AppendGraphicsMessage(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::GRAPHICS, DebugMessage::Type::MESSAGE, severity);
}
void AppendGraphicsWarning(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::GRAPHICS, DebugMessage::Type::WARNING, severity);
}
void AppendGraphicsError(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::GRAPHICS, DebugMessage::Type::ERROR, severity);
}
void AppendPythonMessage(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::PYTHON, DebugMessage::Type::MESSAGE, severity);
}
void AppendPythonWarning(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::PYTHON, DebugMessage::Type::WARNING, severity);
}
void AppendPythonError(const std::string& msg, DebugMessage::Severity severity) {
    AppendDebugMessage(msg, DebugMessage::Source::PYTHON, DebugMessage::Type::ERROR, severity);
}

void PrintDebugLog(int truncateLength) {
    if (isLogEmpty()) {
        std::cout << "[Debug Log is empty]" << std::endl;
        return;
    }
    for (const auto& entry : debugLog) {
        std::string output = entry.ToString();
        if (truncateLength > 0 && output.length() > static_cast<size_t>(truncateLength)) {
            output = output.substr(0, truncateLength) + "...";
        }
        std::cout << output << std::endl;
    }
    if (isLogFull()) {
        std::cout << "[Debug Log is full]" << std::endl;
    }
}
void ClearDebugLog(bool saveBeforeClear) {
    Time time;
    time.update(); // Ensure time is updated for timestamping if needed
    if (saveBeforeClear) SaveDebugLogToFile("debug_log_" + std::to_string(time.lastFrame.time_since_epoch().count()) + ".txt");

    debugLog.clear();
    std::cout << "[Debug Log cleared]" << std::endl;
}
std::vector<DebugMessage> GetDebugLog() {
    return debugLog;
}

bool isLogFull() {
    return getTotalLogSize() >= static_cast<size_t>(MAX_LOG_SIZE);
}
bool willEntryExceedMaxLogSize(int entrySize) {
    return (getTotalLogSize() + static_cast<size_t>(entrySize)) > static_cast<size_t>(MAX_LOG_SIZE);
}
bool isLogEmpty() {
    return debugLog.empty();
}

bool SaveDebugLogToFile(const std::string& filename, const std::string& directory) {
    std::filesystem::path dirPath(directory);
    std::error_code ec;
    if (!std::filesystem::create_directories(dirPath, ec) && ec) {
        AppendError("Failed to create log directory: " + directory + " | " + ec.message());
        return false;
    }

    std::filesystem::path filePath = dirPath / filename;
    std::ofstream outFile(filePath.string(), std::ios::out | std::ios::trunc);
    if (!outFile.is_open()) {
        AppendError("Failed to open file for writing: " + filePath.string());
        return false;
    }
    for (const auto& entry : debugLog) {
        outFile << entry.ToString() << "\n";
    }
    outFile.close();
    return true;
}

bool LoadDebugLogFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        AppendError("Failed to open file for reading: " + filename);
        return false;
    }
    debugLog.clear();
    std::string line;
    while (std::getline(inFile, line)) {
        AppendMessage(line);
    }
    inFile.close();
    return true;
}