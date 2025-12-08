#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <glad/glad.h>

#include "timeClass.h"

#define GL_CHECK(x) do { \
    x; \
    glCheckError_(__FILE__, __LINE__); \
} while (0)

#if DEBUG_MODE
    #define GLCall(x) GL_CHECK(x)
#else
    #define GLCall(x) x
#endif

const long MAX_LOG_SIZE = 1048576; // 1 MB

struct DebugMessage {
    enum Source {
        APPLICATION,
        GRAPHICS,
        PYTHON,
        API,
        WIN_SYS,
        SHADER,
        THIRD_PARTY,
        OTHER_SRC,
        UNKNOWN_SRC
    } source;
    enum Type {
        MESSAGE,
        WARNING,
        ERROR,
        DEPRECATED_BEHAVIOR,
        UNDEFINED_BEHAVIOR,
        PORTABILITY,
        PERFORMANCE,
        MARKER,
        PUSH_GROUP,
        POP_GROUP,
        OTHER_TYP,
        UNKNOWN_TYP
    } type;
    enum Severity {
        LOW,
        MEDIUM,
        HIGH,
        NOTIFICATION,
        UNKNOWN_SEV
    } severity;
    std::string message;
    int count;

    DebugMessage( const std::string& msg, Source src, Type t, Severity sev)
        : source(src), type(t), severity(sev), message(msg), count(1) {}

    std::string ToString() const {
        std::ostringstream oss;
        oss << "[" << SourceToString(source) << "] ["
            << TypeToString(type) << "] "
            << "(Severity: " << SeverityToString(severity) << "): "
            << message;
        if (count > 1) {
            oss << " [" << count << " times]";
        }
        return oss.str();
    }

    static std::string SourceToString(Source src) {
        switch (src) {
            case APPLICATION: return "Application";
            case GRAPHICS: return "Graphics";
            case PYTHON: return "Python";
            case API: return "API";
            case WIN_SYS: return "Window System";
            case SHADER: return "Shader";
            case THIRD_PARTY: return "Third Party";
            case OTHER_SRC: return "Other";
            default: return "Unknown";
        }
    }
    static std::string TypeToString(Type t) {
        switch (t) {
            case MESSAGE: return "Message";
            case WARNING: return "Warning";
            case ERROR: return "Error";
            case DEPRECATED_BEHAVIOR: return "Deprecated Behavior";
            case UNDEFINED_BEHAVIOR: return "Undefined Behavior";
            case PORTABILITY: return "Portability";
            case PERFORMANCE: return "Performance";
            case MARKER: return "Marker";
            case PUSH_GROUP: return "Push Group";
            case POP_GROUP: return "Pop Group";
            case OTHER_TYP: return "Other";
            default: return "Unknown";
        }
    }
    static std::string SeverityToString(Severity sev) {
        switch (sev) {
            case LOW: return "Low";
            case MEDIUM: return "Medium";
            case HIGH: return "High";
            case NOTIFICATION: return "Notification";
            default: return "Unknown";
        }
    }

    // Approximate size including metadata
    int getApproxSize() const {
        return message.size() + std::to_string(count).length() + 50; // overhead for formatting and enum values
    }
};

extern std::vector<DebugMessage> debugLog;

// Check for OpenGL errors
GLenum glCheckError_(const char *file, int line);

// OpenGL debug callback
void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id, GLenum severity, GLsizei length, const char *message, const void *userParam);
void APIENTRY EnableOpenGLDebugger();

void AppendDebugMessage(const DebugMessage& msg, bool saveWhenFull = true);
void AppendDebugMessage(const std::string& msg,DebugMessage::Source source, DebugMessage::Type type, DebugMessage::Severity severity);

void AppendMessage(const std::string& msg, DebugMessage::Severity severity = DebugMessage::LOW);
void AppendWarning(const std::string& warning, DebugMessage::Severity severity = DebugMessage::MEDIUM);
void AppendError(const std::string& error, DebugMessage::Severity severity = DebugMessage::HIGH);
void AppendGraphicsMessage(const std::string& msg, DebugMessage::Severity severity = DebugMessage::LOW);
void AppendGraphicsWarning(const std::string& warning, DebugMessage::Severity severity = DebugMessage::MEDIUM);
void AppendGraphicsError(const std::string& error, DebugMessage::Severity severity = DebugMessage::HIGH);
void AppendPythonMessage(const std::string& msg, DebugMessage::Severity severity = DebugMessage::LOW);
void AppendPythonWarning(const std::string& warning, DebugMessage::Severity severity = DebugMessage::MEDIUM);
void AppendPythonError(const std::string& error, DebugMessage::Severity severity = DebugMessage::HIGH);

void PrintDebugLog(int truncateLength = 1000);
void ClearDebugLog(bool saveBeforeClear = true);

bool isLogFull();
bool willEntryExceedMaxLogSize(int entrySize);
bool isLogEmpty();
bool SaveDebugLogToFile(const std::string& filename, const std::string& directory = "./logs");
bool LoadDebugLogFromFile(const std::string& filename);