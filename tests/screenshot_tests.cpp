#include "screenshot.h"

#include <png.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace {

bool CheckRoundTrip(const std::filesystem::path &path, int width, int height,
                    const std::vector<uint32_t> &pixels) {
  if (!WritePngScreenshot(path.string(), width, height, pixels)) {
    return false;
  }

  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, path.string().c_str())) {
    std::fprintf(stderr, "Failed to read PNG: %s\n", image.message);
    png_image_free(&image);
    return false;
  }
  image.format = PNG_FORMAT_RGBA;
  std::vector<unsigned char> decoded(PNG_IMAGE_SIZE(image));
  bool passed = image.width == static_cast<unsigned>(width) &&
                image.height == static_cast<unsigned>(height);
  if (!png_image_finish_read(&image, nullptr, decoded.data(), 0, nullptr)) {
    std::fprintf(stderr, "Failed to decode PNG: %s\n", image.message);
    passed = false;
  } else if (passed) {
    for (size_t i = 0; i < pixels.size(); ++i) {
      passed = passed &&
               decoded[4 * i] == static_cast<unsigned char>(pixels[i]) &&
               decoded[4 * i + 1] == static_cast<unsigned char>(pixels[i] >> 8) &&
               decoded[4 * i + 2] == static_cast<unsigned char>(pixels[i] >> 16) &&
               decoded[4 * i + 3] == 255;
    }
  }
  png_image_free(&image);
  if (!passed) {
    std::fprintf(stderr, "PNG dimensions, colors, or row order differ\n");
  }
  return passed;
}

} // namespace

int main() {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("happyray-screenshot-test-" +
                          std::to_string(std::chrono::steady_clock::now()
                                             .time_since_epoch().count()));
  if (!std::filesystem::create_directory(directory)) {
    std::fprintf(stderr, "Failed to create screenshot test directory\n");
    return EXIT_FAILURE;
  }
  const auto path = directory / "test.png";
  bool passed = CheckRoundTrip(path, 2, 2,
                              {0xff0000ff, 0xff00ff00, 0xffff0000, 0xffffffff});
  // A second capture at a different, non-square resolution.
  passed = CheckRoundTrip(path, 3, 1, {0xff804020, 0xff123456, 0xffabcdef}) &&
           passed;
  passed = !WritePngScreenshot(path.string(), 0, 1, {}) && passed;
  passed = !WritePngScreenshot(path.string(), -1, 1, {}) && passed;
  passed = !WritePngScreenshot(path.string(), 2, 2, {0}) && passed;
  passed = !WritePngScreenshot((directory / "missing" / "test.png").string(),
                               1, 1, {0xffffffff}) && passed;
  std::filesystem::remove(path);
  std::filesystem::remove(directory);
  if (!passed) {
    std::fprintf(stderr, "Screenshot tests failed\n");
    return EXIT_FAILURE;
  }
  std::printf("Screenshot tests passed\n");
  return EXIT_SUCCESS;
}
