#pragma once

#include <vector>
#include <cstdint>

struct Image
{
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> pixels; // row-major, top-to-bottom or bottom-to-top (document)

    Image() {
        pixels = std::vector<uint8_t>();
    }
};