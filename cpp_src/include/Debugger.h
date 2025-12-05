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
        OPENGL,
        PYTHON
    } source;
    enum Type {
        MESSAGE,
        WARNING,
        ERROR
    } type;
    enum Severity {
        LOW,
        MEDIUM,
        HIGH
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
            oss << " [X" << count << "]";
        }
        return oss.str();
    }

    static std::string SourceToString(Source src) {
        switch (src) {
            case APPLICATION: return "Application";
            case OPENGL: return "OpenGL";
            case PYTHON: return "Python";
            default: return "Unknown";
        }
    }
    static std::string TypeToString(Type t) {
        switch (t) {
            case MESSAGE: return "Message";
            case WARNING: return "Warning";
            case ERROR: return "Error";
            default: return "Unknown";
        }
    }
    static std::string SeverityToString(Severity sev) {
        switch (sev) {
            case LOW: return "Low";
            case MEDIUM: return "Medium";
            case HIGH: return "High";
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
void AppendOpenGLMessage(const std::string& msg, DebugMessage::Severity severity = DebugMessage::LOW);
void AppendOpenGLWarning(const std::string& warning, DebugMessage::Severity severity = DebugMessage::MEDIUM);
void AppendOpenGLError(const std::string& error, DebugMessage::Severity severity = DebugMessage::HIGH);
void AppendPythonMessage(const std::string& msg, DebugMessage::Severity severity = DebugMessage::LOW);
void AppendPythonWarning(const std::string& warning, DebugMessage::Severity severity = DebugMessage::MEDIUM);
void AppendPythonError(const std::string& error, DebugMessage::Severity severity = DebugMessage::HIGH);

void PrintDebugLog(int truncateLength = 1000);
void ClearDebugLog();

bool isLogFull();
bool willEntryExceedMaxLogSize(int entrySize);
bool isLogEmpty();
bool SaveDebugLogToFile(const std::string& filename, const std::string& directory = "./logs");
bool LoadDebugLogFromFile(const std::string& filename);