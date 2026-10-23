// Interactive SDL2 viewer: loads an OBJ scene, flies a pinhole camera around it
// with the keyboard/mouse, and starts with a simple-shaded preview via Renderer
// (src/render/renderer.h), fading into accumulated AO while the camera is
// still. Renderer is parallelized entirely through Thrust; whether that means
// "on the GPU via CUDA" or "in parallel on the CPU via TBB" was decided when
// this binary was built (CMake's HAPPYRAY_THRUST_BACKEND option) -- there's no
// runtime GPU/CPU toggle here, just whichever backend this executable was
// compiled with (shown in the window title). Camera controls
// (movement/look/reset/step-size) match javorkk/happyray's SDLGLApplication
// exactly (see printHelp() below, bound to F1); 'T' saves a PNG screenshot and
// '1'/'2'/'3' select the uniform / two-level (default) / irregular grid.
// These additions aren't part of happyray's own control scheme,
// since happyray doesn't support switching acceleration structures at runtime.
// Brute force (no acceleration structure) is deliberately not offered here --
// it's too slow to stay interactive on anything but the smallest scenes; it
// remains available for correctness validation via tests/renderer_tests.cpp and
// benchmarking via src/app/benchmark.cpp. The window is resizable at runtime
// (dragging an edge reallocates the framebuffer/texture and resets accumulation
// at the new size, see the SDL_WINDOWEVENT_SIZE_CHANGED handling below); it
// starts at an explicit command-line [width] [height] if given, else at the
// scene's camera.cfg's resX/resY if it has one, else at 512x512 (kDefaultWidth/
// kDefaultHeight below). 'c' (PrintCameraConfig below) dumps the current
// resolution alongside the camera's position/orientation/FOV, so a captured
// viewpoint can be pasted back into a camera.cfg and reproduce the exact
// framing it was captured at.
#include <SDL.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

#include "camera.h"
#include "grid_density_search.h"
#include "obj_loader.h"
#include "renderer.h"
#include "sah_cost.h"
#include "scene.h"
#include "screenshot.h"

namespace {

// Fallback resolution when neither the command line nor the scene's camera.cfg
// (see LoadCameraConfig's resX/resY handling) specifies one.
constexpr int kDefaultWidth = 512;
constexpr int kDefaultHeight = 512;

// Picks the right one of SahCalibration's three measured cost-constant pairs (see
// src/accel/sah_cost.h) for the grid density search -- brute force has no SAH cost model.
const happyray::SahCostConstants &SahCostConstantsFor(happyray::AccelType type,
                                                  const happyray::SahCalibration &calibration) {
  switch (type) {
  case happyray::AccelType::kUniformGrid:
    return calibration.uniform;
  case happyray::AccelType::kTwoLevelGrid:
    return calibration.two_level;
  case happyray::AccelType::kIrregularGrid:
    return calibration.irregular;
  default:
    throw std::logic_error("Density search requires an active grid");
  }
}

uint32_t ToPackedRgba(const happyray::Vec3 &color) {
  auto to_byte = [](float c) {
    const float clamped = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
    return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
  };
  const uint8_t r = to_byte(color.x);
  const uint8_t g = to_byte(color.y);
  const uint8_t b = to_byte(color.z);
  return (255u << 24) | (static_cast<uint32_t>(b) << 16) |
         (static_cast<uint32_t>(g) << 8) | r;
}

// Debug/verification helper: dumps a resolved (post-accumulation-average) pixel
// buffer as a binary PPM. Enabled by setting HAPPYRAY_SCREENSHOT_PATH and
// HAPPYRAY_SCREENSHOT_FRAME in the environment, so the renderer can be checked
// visually (or diffed) without a human driving the interactive window.
void WritePpmScreenshot(const std::string &path, int width, int height,
                        const std::vector<happyray::Vec3> &resolved) {
  std::FILE *file = std::fopen(path.c_str(), "wb");
  if (!file) {
    std::fprintf(stderr, "WritePpmScreenshot: failed to open '%s'\n",
                 path.c_str());
    return;
  }
  std::fprintf(file, "P6\n%d %d\n255\n", width, height);
  for (int i = 0; i < width * height; ++i) {
    const happyray::Vec3 &c = resolved[i];
    auto to_byte = [](float v) {
      const float clamped = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
      return static_cast<unsigned char>(clamped * 255.0f + 0.5f);
    };
    const unsigned char rgb[3] = {to_byte(c.x), to_byte(c.y), to_byte(c.z)};
    std::fwrite(rgb, 1, 3, file);
  }
  std::fclose(file);
  std::printf("Wrote screenshot to '%s'\n", path.c_str());
}

// Prints the current control scheme, matching javorkk/happyray's own
// SDLGLApplication:: printHelp() (bound to F1 there too).
void PrintControlsHelp() {
  std::fprintf(
      stderr,
      "----------------------------------------------------------\n"
      "Controls\n"
      "----------------------------------------------------------\n"
      "Start camera rotation: left click with the mouse inside the window\n"
      "                       (click again to release the mouse)\n"
      "Move left/right/forward/back: a/d/w/s\n"
      "Move up/down: q/z\n"
      "Adjust movement step size: [ / ]\n"
      "Adjust mouse look sensitivity: mouse wheel\n"
      "Reset camera: Space\n"
      "Output camera parameters: c\n"
      "Save PNG screenshot: t\n"
      "Acceleration structure: 1 = uniform, 2 = two-level, 3 = irregular\n"
      "Print this help: F1\n"
      "Quit: Esc\n");
}

// Prints the current camera's parameters in the same happyray camera.cfg text
// format LoadCameraConfig (scene.h) reads, matching javorkk/happyray's
// own 'c' key (there it writes a file; here it's simpler to just print, since
// this project has no interactive save dialog). Includes the current window
// resolution (resX/resY) so a dumped config reproduces the exact viewpoint
// *and* framing it was captured at, not just the camera's orientation.
void PrintCameraConfig(const happyray::Camera &camera, int width, int height) {
  const happyray::Vec3 forward = camera.Forward();
  std::printf("#camera parameters\n");
  std::printf("position\t\t%g %g %g\n", camera.position.x, camera.position.y,
              camera.position.z);
  std::printf("orientation\t\t%g %g %g\n", forward.x, forward.y, forward.z);
  std::printf("up\t\t\t0 1 0\n");
  std::printf("resX\t\t\t%d\n", width);
  std::printf("resY\t\t\t%d\n", height);
  std::printf("FOV\t\t\t%g\n", camera.vertical_fov_deg);
}

// Horizontal FOV derived from a vertical FOV and an aspect ratio, the same
// formula GenerateRay (camera.h) itself uses -- kept in sync so yaw's
// degrees-per-pixel (see the mouse-look block in main()) always matches
// what's actually on screen, not an approximation. Recomputed whenever the
// window is resized, since it depends on width/height.
float ComputeHorizontalFovDeg(float vertical_fov_deg, int width, int height) {
  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  return 2.0f *
         std::atan(aspect * std::tan(0.5f * vertical_fov_deg *
                                     (3.14159265358979323846f / 180.0f))) *
         (180.0f / 3.14159265358979323846f);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <scene.obj> [width] [height]\n", argv[0]);
    return EXIT_FAILURE;
  }
  // Resolution precedence: an explicit command-line [width] [height] wins
  // outright; otherwise the scene's camera.cfg (if any) can specify its own
  // resX/resY (see LoadCameraConfig, scene.h); failing both, fall back
  // to kDefaultWidth/kDefaultHeight. The window is resizable at runtime
  // regardless (see the SDL_WINDOWEVENT_SIZE_CHANGED handling below) -- this
  // only picks the resolution it *starts* at.
  const bool width_from_args = argc > 2;
  const bool height_from_args = argc > 3;
  int width = width_from_args ? std::atoi(argv[2]) : kDefaultWidth;
  int height = height_from_args ? std::atoi(argv[3]) : kDefaultHeight;

  happyray::Scene scene;
  if (!happyray::LoadObj(argv[1], &scene)) {
    return EXIT_FAILURE;
  }
  std::printf("Loaded '%s': %u triangle(s), %u material(s)\n", argv[1],
              scene.TriangleCount(),
              static_cast<unsigned>(scene.materials.size()));
  const happyray::FlatScene flat_scene = happyray::FlattenScene(scene);

  happyray::Camera camera = happyray::MakeDefaultCamera(scene);
  const std::string camera_config_path = happyray::CameraConfigPathFor(argv[1]);
  int config_width = 0;
  int config_height = 0;
  if (happyray::LoadCameraConfig(camera_config_path, &camera, &config_width,
                             &config_height)) {
    std::printf("Loaded camera viewpoint from '%s'\n",
                camera_config_path.c_str());
    if (!width_from_args && config_width > 0) {
      width = config_width;
    }
    if (!height_from_args && config_height > 0) {
      height = config_height;
    }
  }

  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }

  SDL_Window *window =
      SDL_CreateWindow("HappyRay", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                       width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
  if (!window) {
    std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }
  // Accelerated rendering sets thread affinites and locks the process to only use 8 threads.
  // SDL_Renderer *sdl_renderer =
  //     SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  SDL_Renderer *sdl_renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
  if (!sdl_renderer)
  {
    std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }
  // Recreated (destroy + re-create at the new size) whenever the window is
  // resized -- see the SDL_WINDOWEVENT_SIZE_CHANGED handling below.
  SDL_Texture *texture =
      SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_RGBA32,
                        SDL_TEXTUREACCESS_STREAMING, width, height);
  if (!texture) {
    std::fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
    return EXIT_FAILURE;
  }

  happyray::AOIntegrator ao_integrator;
  ao_integrator.max_distance = happyray::DefaultMaxDistance(scene);

  happyray::SimpleIntegrator simple_integrator;

  happyray::Renderer renderer(flat_scene);
  renderer.ResizeFramebuffer(width, height);

  // Acceleration structures (see src/accel); '1'/'2'/'3' select the uniform
  // grid, the two-level grid (the default), and the irregular grid at
  // runtime -- brute force is excluded from this viewer entirely (see this
  // file's header comment). HAPPYRAY_ACCEL_TYPE=uniform_grid, two_level_grid, or
  // irregular_grid selects a different one from startup, for headless
  // verification.
  renderer.BuildUniformGrid();
  renderer.BuildTwoLevelGrid();
  renderer.BuildIrregularGrid();
  happyray::AccelType accel_type = happyray::AccelType::kTwoLevelGrid;
  const char *initial_accel_type = std::getenv("HAPPYRAY_ACCEL_TYPE");
  if (initial_accel_type && std::string(initial_accel_type) == "uniform_grid") {
    accel_type = happyray::AccelType::kUniformGrid;
  } else if (initial_accel_type &&
             std::string(initial_accel_type) == "two_level_grid") {
    accel_type = happyray::AccelType::kTwoLevelGrid;
  } else if (initial_accel_type &&
             std::string(initial_accel_type) == "irregular_grid") {
    accel_type = happyray::AccelType::kIrregularGrid;
  }
  renderer.SetAccelType(accel_type);

  std::array<happyray::GridDensities, 4> best_densities{};
  best_densities[static_cast<size_t>(happyray::AccelType::kUniformGrid)] = {
      happyray::UniformGrid::kDefaultDensity, 0};
  best_densities[static_cast<size_t>(happyray::AccelType::kTwoLevelGrid)] = {
      happyray::TwoLevelGrid::kDefaultTopLevelDensity,
      happyray::TwoLevelGrid::kDefaultSecondLevelDensity};
  best_densities[static_cast<size_t>(happyray::AccelType::kIrregularGrid)] = {
      happyray::IrregularGrid::kDefaultTopLevelDensity,
      happyray::IrregularGrid::kDefaultSecondLevelDensity};

  happyray::GridDensitySearch density_search;
  constexpr uint32_t kMaxPrimitivesForDensitySearch = 1000000;
  const bool density_search_enabled =
      scene.TriangleCount() <= kMaxPrimitivesForDensitySearch;
  happyray::SahCalibration sah_calibration;
  if (density_search_enabled) {
    // Runs a handful of timed micro-benchmarks (see src/accel/sah_cost.h) once, up front, so
    // every later candidate rebuild can be evaluated instantly from its own cell layout alone.
    sah_calibration = happyray::MeasureSahCalibration();
    happyray::PrintSahCalibration(sah_calibration);
    density_search.Reset(accel_type,
                         renderer.ComputeCurrentGridSahCost(
                             scene.WorldBounds(),
                             SahCostConstantsFor(accel_type, sah_calibration)));
  } else {
    std::printf("Grid density search disabled for scenes with more than %u "
                "primitives\n",
                kMaxPrimitivesForDensitySearch);
  }

  std::vector<uint32_t> pixel_buffer(static_cast<size_t>(width) * height);

  const happyray::Camera initial_camera =
      camera; // For the "reset camera" (Space) control.
  bool looking = false;
  int pre_look_mouse_x = width / 2;
  int pre_look_mouse_y = height / 2;
  bool running = true;
  uint32_t last_ticks = SDL_GetTicks();

  // Matches javorkk/happyray's SDLGLApplication exactly: initial step is 5% of
  // the scene's bounding-box diagonal, halved/doubled by '['/']', clamped to
  // its same [0.01, 200] bounds.
  float move_step = happyray::Length(scene.WorldBounds().Extent()) * 0.05f;
  constexpr float kMinStepSize = 0.01f;
  constexpr float kMaxStepSize = 200.0f;

  // Mouse-look sensitivity, expressed as a unitless multiplier rather than a
  // fixed degrees-per-pixel constant: each axis's degrees-per-pixel is
  // derived from that axis's own FOV divided by its own pixel count (yaw from
  // the horizontal FOV over the window's width, pitch from the vertical FOV
  // over its height -- matching GenerateRay's own aspect-ratio handling
  // exactly, see below), so a full window-spanning mouse swipe always rotates
  // by the same fraction of the visible FOV on that axis, regardless of
  // window size or display DPI (a fixed raw-pixel coefficient instead gets
  // more or less sensitive as the window/display's pixels-per-physical-inch
  // changes, which previously made "just lower the constant" an unreliable
  // fix). Adjustable live with the mouse wheel (scroll up/down to
  // increase/decrease) so it doesn't need a recompile to tune, and seeded
  // from HAPPYRAY_LOOK_SENSITIVITY so a comfortable value can be kept across
  // runs.
  const char *initial_look_sensitivity_str =
      std::getenv("HAPPYRAY_LOOK_SENSITIVITY");
  float look_sensitivity = initial_look_sensitivity_str
                               ? std::atof(initial_look_sensitivity_str)
                               : 0.1f;
  constexpr float kMinLookSensitivity = 0.001f;
  constexpr float kMaxLookSensitivity = 5.0f;

  // Separate multiplier applied only to the pitch (vertical-look) axis, on
  // top of look_sensitivity above -- fixed below 1 so yaw (side-to-side) is
  // the dominant/faster rotation, matching how most FPS-style cameras feel
  // (pitch only ever sweeps a bounded +-89 degree range, while yaw sweeps
  // the full, unbounded 360 degrees, so it doesn't need as much per-pixel
  // precision). Not user-adjustable (unlike look_sensitivity above) to keep
  // the control scheme simple; 0.256 was picked by hand as a good default.
  constexpr float kPitchScale = 0.256f;

  // Recomputed on every resize (see SDL_WINDOWEVENT_SIZE_CHANGED below) since
  // it depends on width/height.
  float horizontal_fov_deg =
      ComputeHorizontalFovDeg(camera.vertical_fov_deg, width, height);

  // Optional headless verification hook (see WritePpmScreenshot above): if set,
  // dump a screenshot after the requested number of accumulated frames and exit
  // automatically.
  const char *screenshot_path = std::getenv("HAPPYRAY_SCREENSHOT_PATH");
  const char *screenshot_frame_str = std::getenv("HAPPYRAY_SCREENSHOT_FRAME");
  const int screenshot_at_frame =
      screenshot_frame_str ? std::atoi(screenshot_frame_str) : -1;
  bool screenshot_failed = false;

  while (running) {
    const auto frame_start = std::chrono::steady_clock::now();
    double build_ms = -1;
    bool camera_dirty = false;
    bool screenshot_requested = false;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      } else if (event.type == SDL_KEYDOWN &&
                 event.key.keysym.sym == SDLK_ESCAPE) {
        running = false;
      } else if (event.type == SDL_WINDOWEVENT &&
                 event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        // Fires for both user-driven resizes (dragging an edge/corner, since
        // the window was created with SDL_WINDOW_RESIZABLE) and programmatic
        // ones; data1/data2 are the new size in pixels.
        const int new_width = event.window.data1;
        const int new_height = event.window.data2;
        if (new_width > 0 && new_height > 0 &&
            (new_width != width || new_height != height)) {
          width = new_width;
          height = new_height;
          SDL_DestroyTexture(texture);
          texture =
              SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_RGBA32,
                                SDL_TEXTUREACCESS_STREAMING, width, height);
          if (!texture) {
            std::fprintf(stderr, "SDL_CreateTexture failed after resize: %s\n",
                         SDL_GetError());
            running = false;
          } else {
            pixel_buffer.assign(static_cast<size_t>(width) * height, 0u);
            renderer.ResizeFramebuffer(
                width, height); // Also resets accumulation to 0 samples.
            camera_dirty = true;
            horizontal_fov_deg =
                ComputeHorizontalFovDeg(camera.vertical_fov_deg, width, height);
          }
        }
      } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_t &&
                 event.key.repeat == 0) {
        screenshot_requested = true;
      } else if (event.type == SDL_KEYDOWN && event.key.repeat == 0 &&
                 (event.key.keysym.sym == SDLK_1 ||
                  event.key.keysym.sym == SDLK_2 ||
                  event.key.keysym.sym == SDLK_3)) {
        const happyray::AccelType selected =
            event.key.keysym.sym == SDLK_1 ? happyray::AccelType::kUniformGrid
            : event.key.keysym.sym == SDLK_2 ? happyray::AccelType::kTwoLevelGrid
                                             : happyray::AccelType::kIrregularGrid;
        if (selected != accel_type) {
          accel_type = selected;
          renderer.SetAccelType(accel_type);
          camera_dirty = true;
        }
      } else if (event.type == SDL_KEYDOWN &&
                 event.key.keysym.sym == SDLK_SPACE && event.key.repeat == 0) {
        camera = initial_camera; // happyray's "reset camera" control.
        camera_dirty = true;
      } else if (event.type == SDL_KEYDOWN &&
                 (event.key.keysym.sym == SDLK_LEFTBRACKET ||
                  event.key.keysym.sym == SDLK_RIGHTBRACKET) &&
                 event.key.repeat == 0) {
        // happyray's movement-step-size adjustment ('[' halves, ']' doubles).
        move_step = event.key.keysym.sym == SDLK_LEFTBRACKET ? move_step * 0.5f
                                                             : move_step * 2.0f;
        move_step = move_step < kMinStepSize   ? kMinStepSize
                    : move_step > kMaxStepSize ? kMaxStepSize
                                               : move_step;
      } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_c &&
                 event.key.repeat == 0) {
        PrintCameraConfig(
            camera, width,
            height); // happyray's "output camera parameters" control.
      } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_F1 &&
                 event.key.repeat == 0) {
        PrintControlsHelp();
      } else if (event.type == SDL_MOUSEBUTTONDOWN &&
                 event.button.button == SDL_BUTTON_LEFT) {
        // happyray toggles the mouse grab with a click (not a held button).
        looking = !looking;
        // Deliberately NOT using SDL_SetRelativeMouseMode: its cursor
        // confinement/re-centering is implemented via actual OS-level cursor
        // warps on some backends, and under this environment (a
        // remote/forwarded X11 session) that warp was observed to
        // re-center on the wrong point, injecting a constant, consistently
        // -signed "phantom" delta into every real mouse movement no matter
        // which underlying SDL API was used to read it (both
        // SDL_MOUSEMOTION events and SDL_GetRelativeMouseState sit
        // downstream of the same warp machinery). Implementing our own
        // "virtual" relative mouse instead -- hide the cursor, warp it to
        // the window's exact center every frame ourselves, and diff the
        // next absolute position against that known-good center (see the
        // per-frame block below) -- fully bypasses that subsystem.
        SDL_ShowCursor(looking ? SDL_DISABLE : SDL_ENABLE);
        if (looking) {
          SDL_GetMouseState(&pre_look_mouse_x, &pre_look_mouse_y);
          SDL_WarpMouseInWindow(window, width / 2, height / 2);
        } else {
          SDL_WarpMouseInWindow(window, pre_look_mouse_x, pre_look_mouse_y);
        }
      } else if (event.type == SDL_MOUSEWHEEL) {
        // Live-adjustable mouse-look sensitivity (not part of happyray's own
        // control scheme -- its mouse-look has no adjustable sensitivity at
        // all). Scroll up to increase, down to decrease.
        look_sensitivity *= event.wheel.y > 0 ? 1.25f : 0.8f;
        look_sensitivity =
            look_sensitivity < kMinLookSensitivity   ? kMinLookSensitivity
            : look_sensitivity > kMaxLookSensitivity ? kMaxLookSensitivity
                                                     : look_sensitivity;
        std::printf("Look sensitivity: %.4g (set HAPPYRAY_LOOK_SENSITIVITY=%.4g "
                    "to start at this value next time)\n",
                    look_sensitivity, look_sensitivity);
      }
    }

    const uint32_t now_ticks = SDL_GetTicks();
    const float dt = (now_ticks - last_ticks) / 1000.0f;
    last_ticks = now_ticks;

    if (looking) {
      // Manual "virtual relative mouse": diff the current absolute cursor
      // position against the window's exact center (where we last warped it
      // to -- see the toggle handler and the re-center below), then warp it
      // back to center again so every frame starts from the same known-good
      // reference point. This deliberately bypasses SDL_SetRelativeMouseMode
      // entirely -- see the toggle handler above for why.
      int dx, dy;
      SDL_GetMouseState(&dx, &dy);
      dx -= width / 2;
      dy -= height / 2;
      if (dx != 0 || dy != 0) {
        // Yaw from the horizontal FOV over width, pitch from the vertical
        // FOV over height -- see kPitchScale/horizontal_fov_deg's
        // declarations above for why these are no longer the same rate.
        const float yaw_degrees_per_pixel =
            (horizontal_fov_deg / width) * look_sensitivity;
        const float pitch_degrees_per_pixel =
            (camera.vertical_fov_deg / height) * look_sensitivity * kPitchScale;
        camera.Look(dx * yaw_degrees_per_pixel, -dy * pitch_degrees_per_pixel);
        camera_dirty = true;
      }
      SDL_WarpMouseInWindow(window, width / 2, height / 2);
    }

    // Frame-rate-independent (move_step is a "units/second" rate here, scaled
    // by dt) -- happyray instead moves by move_step every held-key *frame* (so
    // its speed is tied to its own render framerate); HappyRay already had the
    // better-behaved dt-scaled continuous movement, so that part is kept, just
    // re-based on happyray's move_step/['/']' scheme instead of HappyRay's previous
    // ad hoc Shift-to-move-faster speed.
    const Uint8 *keys = SDL_GetKeyboardState(nullptr);
    const float speed = move_step * dt;
    if (keys[SDL_SCANCODE_W]) {
      camera.MoveForward(speed);
      camera_dirty = true;
    }
    if (keys[SDL_SCANCODE_S]) {
      camera.MoveForward(-speed);
      camera_dirty = true;
    }
    if (keys[SDL_SCANCODE_D]) {
      camera.MoveRight(speed);
      camera_dirty = true;
    }
    if (keys[SDL_SCANCODE_A]) {
      camera.MoveRight(-speed);
      camera_dirty = true;
    }
    if (keys[SDL_SCANCODE_Q]) {
      camera.MoveUp(speed);
      camera_dirty = true;
    }
    if (keys[SDL_SCANCODE_Z]) {
      camera.MoveUp(-speed);
      camera_dirty = true;
    }

    if (camera_dirty) {
      renderer.ResetAccumulation();
      if (density_search_enabled) {
        density_search.Reset(accel_type,
                             renderer.ComputeCurrentGridSahCost(
                                 scene.WorldBounds(),
                                 SahCostConstantsFor(accel_type, sah_calibration)));
      }
    }

    simple_integrator.light_direction = camera.Forward() * -1.0f;
    renderer.RenderProgressiveFrame(camera, ao_integrator, simple_integrator);

    if (density_search_enabled && !camera_dirty && running &&
        density_search.CanRebuild(renderer.SampleCount())) {
      auto rebuild_grid = [&](happyray::GridDensities densities) {
        const auto build_start = std::chrono::steady_clock::now();
        renderer.RebuildGridDensities(densities);
        build_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - build_start)
                       .count();
      };
      const happyray::GridDensities candidate = density_search.Candidate();
      rebuild_grid(candidate);
      // No rays cast, no frame rendered, no waiting for interactive timing to accumulate --
      // the candidate's SAH cost (src/accel/sah_cost.h) is available the instant it's built.
      const float candidate_sah_cost = renderer.ComputeCurrentGridSahCost(
          scene.WorldBounds(), SahCostConstantsFor(accel_type, sah_calibration));
      const bool improved = density_search.Evaluate(candidate_sah_cost);
      if (improved) {
        best_densities[static_cast<size_t>(accel_type)] = candidate;
      } else {
        rebuild_grid(best_densities[static_cast<size_t>(accel_type)]);
      }
      if (density_search.Stopped()) {
        const happyray::GridDensities best =
            best_densities[static_cast<size_t>(accel_type)];
        if (accel_type == happyray::AccelType::kUniformGrid) {
          std::printf("Final densities [uniform]: density=%.6g\n",
                      best.top_level);
        } else {
          const char *grid = accel_type == happyray::AccelType::kTwoLevelGrid
                                 ? "two-level"
                                 : "irregular";
          std::printf("Final densities [%s]: top=%.6g leaf=%.6g\n", grid,
                      best.top_level, best.second_level);
        }
        std::fflush(stdout);
      }
      // Search/rebuild time is not camera-motion time.
      last_ticks = SDL_GetTicks();
    }
    const std::vector<happyray::Vec3> &resolved = renderer.ResolvedPixels();

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        pixel_buffer[static_cast<size_t>(y) * width + x] =
            ToPackedRgba(resolved[static_cast<size_t>(y) * width + x]);
      }
    }
    if (screenshot_requested) {
      const auto now = std::chrono::system_clock::now();
      const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
      const std::tm *local_time = std::localtime(&timestamp);
      char date_time[32];
      if (!local_time ||
          std::strftime(date_time, sizeof(date_time), "%Y-%m-%d_%H-%M-%S",
                        local_time) == 0) {
        std::fprintf(stderr, "Failed to format screenshot date/time\n");
      } else {
        const auto microseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()).count() % 1000000;
        char path[80];
        std::snprintf(path, sizeof(path), "happyray_%s-%06lld.png", date_time,
                      static_cast<long long>(microseconds));
        WritePngScreenshot(path, width, height, pixel_buffer);
      }
      // Disk I/O time must not become camera-motion time on the next frame.
      last_ticks = SDL_GetTicks();
    }
    if (screenshot_path &&
        static_cast<int>(renderer.SampleCount()) >= screenshot_at_frame) {
      const std::string path(screenshot_path);
      if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".png") == 0) {
        screenshot_failed = !WritePngScreenshot(path, width, height, pixel_buffer);
      } else {
        WritePpmScreenshot(path, width, height, resolved);
      }
      running = false;
    }
    SDL_UpdateTexture(texture, nullptr, pixel_buffer.data(),
                      width * sizeof(uint32_t));
    SDL_RenderClear(sdl_renderer);
    SDL_RenderCopy(sdl_renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(sdl_renderer);

    char title[256];
    // Brute force is excluded from this viewer (see this file's header
    // comment), so accel_type is always one of these three.
    const char *accel_name =
        accel_type == happyray::AccelType::kUniformGrid    ? "ugrid"
        : accel_type == happyray::AccelType::kTwoLevelGrid ? "2grid"
                                                       : "igrid";
    const double frame_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      frame_start)
            .count();
    const double fps = 1.0 / frame_seconds;
    const int title_length = std::snprintf(
        title, sizeof(title), "HappyRay [%s/%s] - %u spp - %.1f fps",
        happyray::RendererBackendName(), accel_name, renderer.SampleCount(), fps);
    if (build_ms >= 0 && title_length >= 0 &&
        static_cast<size_t>(title_length) < sizeof(title)) {
      std::snprintf(title + title_length, sizeof(title) - title_length,
                    " - build: %.2f ms", build_ms);
    }
    SDL_SetWindowTitle(window, title);
  }

  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(sdl_renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return screenshot_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
