#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Pixels use the viewer's packed 0xAABBGGRR representation, in top-to-bottom order.
bool WritePngScreenshot(const std::string &path, int width, int height,
                        const std::vector<uint32_t> &pixels);
