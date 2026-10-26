// SDL2 smoke test: proves that an SDL2 window/renderer/texture can be created and a pixel
// buffer presented, all from a single CMake-built executable. The pixel buffer itself is
// computed on the CPU (a plain RGBA gradient) -- this file has no CUDA/Thrust dependency at
// all, so (unlike the project's actual renderer) it builds and runs identically regardless of
// which Thrust backend (CUDA or TBB) the rest of the project is configured with.
#include <SDL.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWidth = 256;
constexpr int kHeight = 256;

void RenderGradient(std::vector<uint32_t>* out_pixels) {
    out_pixels->resize(static_cast<size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const uint8_t r = static_cast<uint8_t>(255.0f * x / (kWidth - 1));
            const uint8_t g = static_cast<uint8_t>(255.0f * y / (kHeight - 1));
            const uint8_t b = 128;
            const uint8_t a = 255;
            (*out_pixels)[y * kWidth + x] = (a << 24) | (b << 16) | (g << 8) | r;
        }
    }
}

bool DisplayPixelsWithSdl(const std::vector<uint32_t>& pixels) {
    if (std::getenv("WSL_DISTRO_NAME") && !std::getenv("SDL_VIDEODRIVER")) {
        setenv("SDL_VIDEODRIVER", "dummy", /*overwrite=*/0);
    }

    // Prefer a real display; fall back to the headless "dummy" driver so the smoke test still
    // exercises the SDL2 rendering path when no X server is reachable (e.g. in CI).
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init with default driver failed (%s); retrying with dummy\n",
                     SDL_GetError());
        setenv("SDL_VIDEODRIVER", "dummy", /*overwrite=*/1);
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::fprintf(stderr, "SDL_Init with dummy driver failed: %s\n", SDL_GetError());
            return false;
        }
    }

    SDL_Window* window = SDL_CreateWindow("HappyRay SDL2 smoke test", SDL_WINDOWPOS_CENTERED,
                                           SDL_WINDOWPOS_CENTERED, kWidth, kHeight,
                                           SDL_WINDOW_HIDDEN);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return false;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    if (!renderer) {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return false;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                              SDL_TEXTUREACCESS_STREAMING, kWidth, kHeight);
    if (!texture) {
        std::fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return false;
    }

    SDL_UpdateTexture(texture, nullptr, pixels.data(), kWidth * sizeof(uint32_t));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return true;
}

}  // namespace

int main() {
    std::vector<uint32_t> pixels;
    RenderGradient(&pixels);

    // Sanity-check a few pixel values before handing them to SDL2.
    const uint32_t top_left = pixels[0];
    const uint32_t bottom_right = pixels[(kHeight - 1) * kWidth + (kWidth - 1)];
    if ((top_left & 0xFF) != 0 || (bottom_right & 0xFF) != 255) {
        std::fprintf(stderr, "Unexpected gradient values: top_left=%08x bottom_right=%08x\n",
                     top_left, bottom_right);
        return EXIT_FAILURE;
    }

    if (!DisplayPixelsWithSdl(pixels)) {
        return EXIT_FAILURE;
    }

    std::printf("HappyRay SDL2 smoke test passed.\n");
    return EXIT_SUCCESS;
}
