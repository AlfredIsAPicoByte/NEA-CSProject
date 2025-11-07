#include "colorClass.h"

Color::Color(float r, float g, float b, float a)
{
    this->r = r;
    this->g = g;
    this->b = b;
    this->a = a;
}
Color::Color(std::string hex)
{
    setHex(hex);
}

void Color::setRGB(float r, float g, float b, float a)
{
    this->r = r;
    this->g = g;
    this->b = b;
    this->a = a;
}

void Color::setHex(std::string hex)
{
    if (hex.empty() || hex[0] != '#') return;
    hex = hex.substr(1);

    if (hex.length() == 3)
    {
        r = float(std::stoi(std::string(2, hex[0]), nullptr, 16)) / 255.0f;
        g = float(std::stoi(std::string(2, hex[1]), nullptr, 16)) / 255.0f;
        b = float(std::stoi(std::string(2, hex[2]), nullptr, 16)) / 255.0f;
        a = 1.0f;
    }
    else if (hex.length() == 6)
    {
        r = float(std::stoi(hex.substr(0, 2), nullptr, 16)) / 255.0f;
        g = float(std::stoi(hex.substr(2, 2), nullptr, 16)) / 255.0f;
        b = float(std::stoi(hex.substr(4, 2), nullptr, 16)) / 255.0f;
        a = 1.0f;
    }
    else if (hex.length() == 8)
    {
        r = float(std::stoi(hex.substr(0, 2), nullptr, 16)) / 255.0f;
        g = float(std::stoi(hex.substr(2, 2), nullptr, 16)) / 255.0f;
        b = float(std::stoi(hex.substr(4, 2), nullptr, 16)) / 255.0f;
        a = float(std::stoi(hex.substr(6, 2), nullptr, 16)) / 255.0f;
    }
    else
    {
        // Invalid format, set to black and fully opaque
        r = g = b = 0.0f;
        a = 1.0f;
    }
}

std::string Color::toHex() const
{
    char hexCol[9];
    snprintf(hexCol, sizeof(hexCol), "#%02X%02X%02X%02X",
        static_cast<int>(r * 255),
        static_cast<int>(g * 255),
        static_cast<int>(b * 255),
        static_cast<int>(a * 255));
    return std::string(hexCol);
}

void Color::print() const
{
    std::cout << "Color(r: " << r << ", g: " << g << ", b: " << b << ", a: " << a << ")" << std::endl;
}

Color Color::lerp(const Color& start, const Color& end, float t)
{
    t = clamp(t, 0.0f, 1.0f);
    return Color(
        start.r + (end.r - start.r) * t,
        start.g + (end.g - start.g) * t,
        start.b + (end.b - start.b) * t,
        start.a + (end.a - start.a) * t
    );
}

