#include "screenshot.h"

#include <png.h>

#include <cstdio>
#include <limits>

bool WritePngScreenshot(const std::string &path, int width, int height,
                        const std::vector<uint32_t> &pixels) {
  if (width <= 0 || height <= 0 ||
      static_cast<size_t>(width) >
          std::numeric_limits<size_t>::max() / static_cast<size_t>(height) ||
      static_cast<size_t>(width) * height != pixels.size() ||
      pixels.size() > std::numeric_limits<size_t>::max() / 3) {
    std::fprintf(stderr, "WritePngScreenshot: invalid framebuffer for '%s'\n",
                 path.c_str());
    return false;
  }

  std::vector<unsigned char> rgb(pixels.size() * 3);
  for (size_t i = 0; i < pixels.size(); ++i) {
    rgb[3 * i] = static_cast<unsigned char>(pixels[i]);
    rgb[3 * i + 1] = static_cast<unsigned char>(pixels[i] >> 8);
    rgb[3 * i + 2] = static_cast<unsigned char>(pixels[i] >> 16);
  }

  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = static_cast<png_uint_32>(width);
  image.height = static_cast<png_uint_32>(height);
  image.format = PNG_FORMAT_RGB;
  const bool success =
      png_image_write_to_file(&image, path.c_str(), 0, rgb.data(), 0, nullptr) != 0;
  if (!success) {
    std::fprintf(stderr, "WritePngScreenshot: failed to write '%s': %s\n",
                 path.c_str(), image.message);
  } else {
    std::printf("Wrote screenshot to '%s'\n", path.c_str());
  }
  png_image_free(&image);
  return success;
}
