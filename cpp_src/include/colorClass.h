#pragma once

#include <glad/glad.h>
#include <string>
#include <iostream>

class Color
{
public:
    float r, g, b, a;

    Color() : r(0), g(0), b(0), a(1) {}
    Color(float red, float green, float blue, float alpha = 1.0f);
    Color(std::string hex);

    void setRGB(float red, float green, float blue, float alpha = 1.0f);
    void setHex(std::string hex);

    std::string toHex() const;
    void print() const;

    static Color lerp(const Color& start, const Color& end, float t);

private:
    int hexToInt(char c) const
    {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1; // Invalid character
    }
    static float clamp(float value, float min, float max)
    {
        return (value < min) ? min : (value > max) ? max : value;
    }
};

void applyClearColor(const Color& color);