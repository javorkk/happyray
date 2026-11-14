// Interactive SDL2 viewer. Camera controls follow javorkk/happyray's SDLGLApplication (F1 prints
// them). Brute force is not offered since it is too slow to be interactive.
#include <SDL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "../accel/sah_cost.h"
#include "../render/camera.h"
#include "../render/renderer.h"
#include "animation.h"
#include "attractor_scene.h"
#include "grid_density_search.h"
#include "obj_loader.h"
#include "scene.h"
#include "screenshot.h"

namespace
{

	constexpr int kDefaultWidth	 = 512;
	constexpr int kDefaultHeight = 512;

	const happyray::SahCostConstants &SahCostConstantsFor(happyray::AccelType			  type,
														  const happyray::SahCalibration &calibration)
	{
		switch (type)
		{
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

	uint32_t ToPackedRgba(const happyray::Vec3f &color)
	{
		auto to_byte = [](float c)
		{
			const float clamped = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
			return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
		};
		const uint8_t r = to_byte(color.x);
		const uint8_t g = to_byte(color.y);
		const uint8_t b = to_byte(color.z);
		return (255u << 24) | (static_cast<uint32_t>(b) << 16) | (static_cast<uint32_t>(g) << 8) | r;
	}

	// Headless verification: HAPPYRAY_SCREENSHOT_PATH/HAPPYRAY_SCREENSHOT_FRAME dump a PPM.
	void
	WritePpmScreenshot(const std::string &path, int width, int height, const std::vector<happyray::Vec3f> &resolved)
	{
		std::FILE *file = std::fopen(path.c_str(), "wb");
		if (!file)
		{
			std::fprintf(stderr, "WritePpmScreenshot: failed to open '%s'\n", path.c_str());
			return;
		}
		std::fprintf(file, "P6\n%d %d\n255\n", width, height);
		for (int i = 0; i < width * height; ++i)
		{
			const happyray::Vec3f &c	   = resolved[i];
			auto				   to_byte = [](float v)
			{
				const float clamped = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
				return static_cast<unsigned char>(clamped * 255.0f + 0.5f);
			};
			const unsigned char rgb[3] = {to_byte(c.x), to_byte(c.y), to_byte(c.z)};
			std::fwrite(rgb, 1, 3, file);
		}
		std::fclose(file);
		std::printf("Wrote screenshot to '%s'\n", path.c_str());
	}

	void PrintControlsHelp()
	{
		std::fprintf(stderr,
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
					 "Toggle collision detection: o\n"
					 "Save PNG screenshot: t\n"
					 "Acceleration structure: 1 = uniform, 2 = two-level, 3 = irregular\n"
					 "Pause/resume animation: p\n"
					 "Print this help: F1\n"
					 "Quit: Esc\n");
	}

	// Prints happyray's camera.cfg format (including resX/resY) for pasting back into a file.
	void PrintCameraConfig(const happyray::Camera &camera, int width, int height)
	{
		const happyray::Vec3f forward = camera.Forward();
		std::printf("#camera parameters\n");
		std::printf("position\t\t%g %g %g\n", camera.position.x, camera.position.y, camera.position.z);
		std::printf("orientation\t\t%g %g %g\n", forward.x, forward.y, forward.z);
		std::printf("up\t\t\t0 1 0\n");
		std::printf("resX\t\t\t%d\n", width);
		std::printf("resY\t\t\t%d\n", height);
		std::printf("FOV\t\t\t%g\n", camera.vertical_fov_deg);
	}

	// Same formula as GenerateRay, so yaw's degrees-per-pixel matches the screen.
	float ComputeHorizontalFovDeg(float vertical_fov_deg, int width, int height)
	{
		const float aspect = static_cast<float>(width) / static_cast<float>(height);
		return 2.0f * std::atan(aspect * std::tan(0.5f * vertical_fov_deg * (3.14159265358979323846f / 180.0f))) *
			   (180.0f / 3.14159265358979323846f);
	}

} // namespace

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::fprintf(stderr,
					 "Usage: %s <scene.obj|scene.usd[a|c|z]|animation.cfg|attractor:<points|spheres|curves>[:count]> [width] "
					 "[height]\n",
					 argv[0]);
		return EXIT_FAILURE;
	}
	// Resolution: command line, else the scene's camera.cfg, else the default.
	const bool width_from_args	= argc > 2;
	const bool height_from_args = argc > 3;
	int		   width			= width_from_args ? std::atoi(argv[2]) : kDefaultWidth;
	int		   height			= height_from_args ? std::atoi(argv[3]) : kDefaultHeight;

	happyray::Scene				scene;
	happyray::KeyframeAnimation animation;
	const std::string			input_extension	 = std::filesystem::path(argv[1]).extension().string();
	const bool					is_usd			 = happyray::IsUsdPath(argv[1]);
	bool						animation_loaded = input_extension == ".cfg";
	happyray::PrimitiveType		attractor_type;
	uint32_t					attractor_count = 0;
	if (happyray::ParseAttractorSpec(argv[1], &attractor_type, &attractor_count))
	{
		scene = happyray::MakeAttractorScene(attractor_type, attractor_count);
	}
	else if (animation_loaded)
	{
		if (!animation.LoadConfig(argv[1], &scene))
		{
			std::fprintf(stderr, "Failed to load animation configuration '%s'\n", argv[1]);
			return EXIT_FAILURE;
		}
		animation_loaded = animation.IsLoaded();
	}
	else if (is_usd)
	{
		if (!animation.LoadUsd(argv[1], &scene))
		{
			return EXIT_FAILURE;
		}
		animation_loaded = animation.IsLoaded();
	}
	else if (input_extension == ".obj")
	{
		if (!happyray::LoadObj(argv[1], &scene))
		{
			return EXIT_FAILURE;
		}
	}
	else
	{
		std::fprintf(stderr, "Input must be an OBJ/USD scene or animation CFG file\n");
		return EXIT_FAILURE;
	}
	const std::string camera_config_path =
		animation.CameraConfigPath().empty() ? happyray::CameraConfigPathFor(argv[1]) : animation.CameraConfigPath();
	std::printf("Loaded '%s': %u %s primitive(s), %u material(s)\n",
				argv[1],
				scene.PrimitiveCount(),
				happyray::PrimitiveTypeName(scene.Type()),
				static_cast<unsigned>(scene.materials.size()));
	if (animation_loaded)
	{
		std::printf("Loaded keyframe animation (frame step %.4g per rendered frame)\n", animation.FrameStep());
	}
	const happyray::FlatScene flat_scene =
		animation_loaded ? animation.InterpolatedScene() : happyray::FlattenScene(scene);

	happyray::Camera camera		   = happyray::MakeDefaultCamera(scene);
	int				 config_width  = 0;
	int				 config_height = 0;
	if (happyray::LoadCameraConfig(camera_config_path, &camera, &config_width, &config_height))
	{
		std::printf("Loaded camera viewpoint from '%s'\n", camera_config_path.c_str());
		if (!width_from_args && config_width > 0)
		{
			width = config_width;
		}
		if (!height_from_args && config_height > 0)
		{
			height = config_height;
		}
	}

	if (SDL_Init(SDL_INIT_VIDEO) != 0)
	{
		std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}

	SDL_Window *window = SDL_CreateWindow("HappyRay",
										  SDL_WINDOWPOS_CENTERED,
										  SDL_WINDOWPOS_CENTERED,
										  width,
										  height,
										  SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
	if (!window)
	{
		std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}
	// SDL_RENDERER_ACCELERATED pins the process to 8 threads.
	SDL_Renderer *sdl_renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
	if (!sdl_renderer)
	{
		std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}
	SDL_Texture *texture =
		SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);
	if (!texture)
	{
		std::fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}

	happyray::AOIntegrator ao_integrator;
	ao_integrator.max_distance = happyray::DefaultMaxDistance(scene);

	happyray::SimpleIntegrator simple_integrator;

	happyray::Renderer renderer(flat_scene);
	renderer.ResizeFramebuffer(width, height);
	renderer.UpdateViewDependentGeometry(camera);
	// HAPPYRAY_CELL_OVERLAP=aabb, exact, or hybrid overrides the backend's grid cell assignment.
	if (const char *overlap = std::getenv("HAPPYRAY_CELL_OVERLAP"))
	{
		const std::string mode = overlap;
		if (mode == "aabb")
			renderer.SetCellOverlapMode(happyray::CellOverlapMode::kAabb);
		else if (mode == "exact")
			renderer.SetCellOverlapMode(happyray::CellOverlapMode::kExact);
		else if (mode == "hybrid")
			renderer.SetCellOverlapMode(happyray::CellOverlapMode::kHybrid);
	}
	std::printf("Grid cell overlap: %s\n", happyray::CellOverlapModeName(renderer.GetCellOverlapMode()));

	// HAPPYRAY_ACCEL_TYPE=uniform_grid, two_level_grid, or irregular_grid overrides the default.
	happyray::AccelType accel_type		   = happyray::AccelType::kTwoLevelGrid;
	const char		   *initial_accel_type = std::getenv("HAPPYRAY_ACCEL_TYPE");
	if (initial_accel_type && std::string(initial_accel_type) == "uniform_grid")
	{
		accel_type = happyray::AccelType::kUniformGrid;
	}
	else if (initial_accel_type && std::string(initial_accel_type) == "two_level_grid")
	{
		accel_type = happyray::AccelType::kTwoLevelGrid;
	}
	else if (initial_accel_type && std::string(initial_accel_type) == "irregular_grid")
	{
		accel_type = happyray::AccelType::kIrregularGrid;
	}
	// A paused animation's geometry doesn't change, so per-build detection is skipped.
	bool collisions_enabled			= false;
	auto update_collision_detection = [&]()
	{ renderer.SetCollisionDetectionEnabled(collisions_enabled && (!animation_loaded || !animation.IsPaused())); };
	update_collision_detection();
	renderer.SetAccelType(accel_type);

	std::array<happyray::GridDensities, 4> best_densities{};
	best_densities[static_cast<size_t>(happyray::AccelType::kUniformGrid)]	= {happyray::UniformGrid::kDefaultDensity,
																			   0};
	best_densities[static_cast<size_t>(happyray::AccelType::kTwoLevelGrid)] = {
		happyray::TwoLevelGrid::kDefaultTopLevelDensity,
		happyray::TwoLevelGrid::kDefaultSecondLevelDensity};
	best_densities[static_cast<size_t>(happyray::AccelType::kIrregularGrid)] = {
		happyray::IrregularGrid::kDefaultTopLevelDensity,
		happyray::IrregularGrid::kDefaultSecondLevelDensity};
	renderer.BuildAccelerationStructure({best_densities[static_cast<size_t>(accel_type)]});

	happyray::GridDensitySearch density_search;
	constexpr uint32_t			kMaxPrimitivesForDensitySearch = 1000000;
	const bool					density_search_enabled = scene.PrimitiveCount() <= kMaxPrimitivesForDensitySearch;
	happyray::SahCalibration	sah_calibration;
	if (density_search_enabled)
	{
		sah_calibration = happyray::MeasureSahCalibration();
		happyray::PrintSahCalibration(sah_calibration);
		density_search.Reset(
			accel_type,
			renderer.ComputeCurrentGridSahCost(scene.WorldBounds(), SahCostConstantsFor(accel_type, sah_calibration)));
	}
	else
	{
		std::printf("Grid density search disabled for scenes with more than %u "
					"primitives\n",
					kMaxPrimitivesForDensitySearch);
	}

	std::vector<uint32_t> pixel_buffer(static_cast<size_t>(width) * height);

	const happyray::Camera initial_camera	= camera;
	bool				   looking			= false;
	int					   pre_look_mouse_x = width / 2;
	int					   pre_look_mouse_y = height / 2;
	bool				   running			= true;
	uint32_t			   last_ticks		= SDL_GetTicks();

	// As in happyray: 5% of the scene diagonal, halved/doubled by '['/']', clamped to [0.01, 200].
	float			move_step	 = happyray::Length(scene.WorldBounds().Extent()) * 0.05f;
	constexpr float kMinStepSize = 0.01f;
	constexpr float kMaxStepSize = 200.0f;

	// Degrees per pixel = that axis's FOV / pixel count * sensitivity, so a full-window swipe turns
	// by the same fraction of the view at any window size or DPI. Mouse wheel adjusts it.
	const char	   *initial_look_sensitivity_str = std::getenv("HAPPYRAY_LOOK_SENSITIVITY");
	float			look_sensitivity	= initial_look_sensitivity_str ? std::atof(initial_look_sensitivity_str) : 0.1f;
	constexpr float kMinLookSensitivity = 0.001f;
	constexpr float kMaxLookSensitivity = 5.0f;

	// Pitch is slower than yaw; hand-tuned.
	constexpr float kPitchScale = 0.256f;

	float horizontal_fov_deg = ComputeHorizontalFovDeg(camera.vertical_fov_deg, width, height);

	const char *screenshot_path		 = std::getenv("HAPPYRAY_SCREENSHOT_PATH");
	const char *screenshot_frame_str = std::getenv("HAPPYRAY_SCREENSHOT_FRAME");
	const int	screenshot_at_frame	 = screenshot_frame_str ? std::atoi(screenshot_frame_str) : -1;
	int			rendered_frame_count = 0;
	bool		screenshot_failed	 = false;

	// Excludes the collision detection that ran with the build.
	auto grid_build_ms_since = [&](std::chrono::steady_clock::time_point start)
	{
		const double elapsed_ms =
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		return renderer.CollisionDetectionEnabled() ? std::max(0.0, elapsed_ms - renderer.LastCollisionStats().total_ms)
													: elapsed_ms;
	};

	// Point/curve radii are refreshed only once the camera stops.
	bool view_geometry_stale = false;

	while (running)
	{
		const auto frame_start			= std::chrono::steady_clock::now();
		double	   build_ms				= -1;
		bool	   camera_dirty			= false;
		bool	   screenshot_requested = false;
		SDL_Event  event;
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
			{
				running = false;
			}
			else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
			{
				running = false;
			}
			else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
			{
				const int new_width	 = event.window.data1;
				const int new_height = event.window.data2;
				if (new_width > 0 && new_height > 0 && (new_width != width || new_height != height))
				{
					width  = new_width;
					height = new_height;
					SDL_DestroyTexture(texture);
					texture = SDL_CreateTexture(sdl_renderer,
												SDL_PIXELFORMAT_RGBA32,
												SDL_TEXTUREACCESS_STREAMING,
												width,
												height);
					if (!texture)
					{
						std::fprintf(stderr, "SDL_CreateTexture failed after resize: %s\n", SDL_GetError());
						running = false;
					}
					else
					{
						pixel_buffer.assign(static_cast<size_t>(width) * height, 0u);
						renderer.ResizeFramebuffer(width, height);
						camera_dirty	   = true;
						horizontal_fov_deg = ComputeHorizontalFovDeg(camera.vertical_fov_deg, width, height);
					}
				}
			}
			else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_t && event.key.repeat == 0)
			{
				screenshot_requested = true;
			}
			else if (event.type == SDL_KEYDOWN && event.key.repeat == 0 &&
					 (event.key.keysym.sym == SDLK_1 || event.key.keysym.sym == SDLK_2 ||
					  event.key.keysym.sym == SDLK_3))
			{
				const happyray::AccelType selected = event.key.keysym.sym == SDLK_1 ? happyray::AccelType::kUniformGrid
													 : event.key.keysym.sym == SDLK_2
														 ? happyray::AccelType::kTwoLevelGrid
														 : happyray::AccelType::kIrregularGrid;
				if (selected != accel_type)
				{
					accel_type = selected;
					renderer.SetAccelType(accel_type);
					if (!renderer.HasAccelerationStructure())
						renderer.BuildAccelerationStructure({best_densities[static_cast<size_t>(accel_type)]});
					camera_dirty = true;
				}
			}
			else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_SPACE && event.key.repeat == 0)
			{
				camera		 = initial_camera;
				camera_dirty = true;
			}
			else if (event.type == SDL_KEYDOWN &&
					 (event.key.keysym.sym == SDLK_LEFTBRACKET || event.key.keysym.sym == SDLK_RIGHTBRACKET) &&
					 event.key.repeat == 0)
			{
				move_step = event.key.keysym.sym == SDLK_LEFTBRACKET ? move_step * 0.5f : move_step * 2.0f;
				move_step = move_step < kMinStepSize   ? kMinStepSize
							: move_step > kMaxStepSize ? kMaxStepSize
													   : move_step;
			}
			else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_c && event.key.repeat == 0)
			{
				PrintCameraConfig(camera, width,
								  height);
			}
			else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_F1 && event.key.repeat == 0)
			{
				PrintControlsHelp();
			}
			else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_p && event.key.repeat == 0 &&
					 animation_loaded)
			{
				animation.TogglePause();
				update_collision_detection();
			}
			else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_o && event.key.repeat == 0)
			{
				collisions_enabled = !collisions_enabled;
				update_collision_detection();
				if (collisions_enabled)
				{
					renderer.DetectCollisions();
				}
				else
				{
					renderer.ClearCollisions();
				}
				renderer.ResetAccumulation();
			}
			else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT)
			{
				// happyray toggles the mouse grab with a click.
				looking = !looking;
				// Not SDL_SetRelativeMouseMode: on forwarded X11 its cursor warps re-center on the
				// wrong point and inject a constant phantom delta. We warp to the window center
				// ourselves and diff against it each frame.
				SDL_ShowCursor(looking ? SDL_DISABLE : SDL_ENABLE);
				if (looking)
				{
					SDL_GetMouseState(&pre_look_mouse_x, &pre_look_mouse_y);
					SDL_WarpMouseInWindow(window, width / 2, height / 2);
				}
				else
				{
					SDL_WarpMouseInWindow(window, pre_look_mouse_x, pre_look_mouse_y);
				}
			}
			else if (event.type == SDL_MOUSEWHEEL)
			{
				look_sensitivity *= event.wheel.y > 0 ? 1.25f : 0.8f;
				look_sensitivity = look_sensitivity < kMinLookSensitivity	? kMinLookSensitivity
								   : look_sensitivity > kMaxLookSensitivity ? kMaxLookSensitivity
																			: look_sensitivity;
				std::printf("Look sensitivity: %.4g (set HAPPYRAY_LOOK_SENSITIVITY=%.4g "
							"to start at this value next time)\n",
							look_sensitivity,
							look_sensitivity);
			}
		}

		const uint32_t now_ticks = SDL_GetTicks();
		const float	   dt		 = (now_ticks - last_ticks) / 1000.0f;
		last_ticks				 = now_ticks;

		if (looking)
		{
			// Virtual relative mouse (see the toggle handler): diff against the center, re-center.
			int dx, dy;
			SDL_GetMouseState(&dx, &dy);
			dx -= width / 2;
			dy -= height / 2;
			if (dx != 0 || dy != 0)
			{
				const float yaw_degrees_per_pixel = (horizontal_fov_deg / width) * look_sensitivity;
				const float pitch_degrees_per_pixel =
					(camera.vertical_fov_deg / height) * look_sensitivity * kPitchScale;
				camera.Look(dx * yaw_degrees_per_pixel, -dy * pitch_degrees_per_pixel);
				camera_dirty = true;
			}
			SDL_WarpMouseInWindow(window, width / 2, height / 2);
		}

		// Unlike happyray, movement is scaled by dt so speed is frame-rate independent.
		const Uint8 *keys  = SDL_GetKeyboardState(nullptr);
		const float	 speed = move_step * dt;
		if (keys[SDL_SCANCODE_W])
		{
			camera.MoveForward(speed);
			camera_dirty = true;
		}
		if (keys[SDL_SCANCODE_S])
		{
			camera.MoveForward(-speed);
			camera_dirty = true;
		}
		if (keys[SDL_SCANCODE_D])
		{
			camera.MoveRight(speed);
			camera_dirty = true;
		}
		if (keys[SDL_SCANCODE_A])
		{
			camera.MoveRight(-speed);
			camera_dirty = true;
		}
		if (keys[SDL_SCANCODE_Q])
		{
			camera.MoveUp(speed);
			camera_dirty = true;
		}
		if (keys[SDL_SCANCODE_Z])
		{
			camera.MoveUp(-speed);
			camera_dirty = true;
		}

		if (camera_dirty)
		{
			view_geometry_stale = true;
		}
		else if (view_geometry_stale)
		{
			view_geometry_stale	   = false;
			const auto build_start = std::chrono::steady_clock::now();
			if (renderer.UpdateViewDependentGeometry(camera))
			{
				build_ms	 = grid_build_ms_since(build_start);
				camera_dirty = true;
			}
		}

		if (camera_dirty)
		{
			renderer.ResetAccumulation();
			if (density_search_enabled)
			{
				density_search.Reset(
					accel_type,
					renderer.ComputeCurrentGridSahCost(scene.WorldBounds(),
													   SahCostConstantsFor(accel_type, sah_calibration)));
			}
		}

		simple_integrator.light_direction = camera.Forward() * -1.0f;
		renderer.RenderProgressiveFrame(camera, ao_integrator, simple_integrator);
		++rendered_frame_count;

		const bool density_search_view_stable = !animation_loaded || animation.IsPaused();
		if (density_search_enabled && density_search_view_stable && !camera_dirty && running &&
			density_search.CanRebuild(renderer.SampleCount()))
		{
			auto rebuild_grid = [&](happyray::GridDensities densities)
			{
				const auto build_start = std::chrono::steady_clock::now();
				renderer.RebuildGridDensities(densities);
				build_ms = grid_build_ms_since(build_start);
			};
			const happyray::GridDensities candidate = density_search.Candidate();
			rebuild_grid(candidate);
			const float candidate_sah_cost =
				renderer.ComputeCurrentGridSahCost(scene.WorldBounds(),
												   SahCostConstantsFor(accel_type, sah_calibration));
			const bool improved = density_search.Evaluate(candidate_sah_cost);
			if (improved)
			{
				best_densities[static_cast<size_t>(accel_type)] = candidate;
			}
			else
			{
				rebuild_grid(best_densities[static_cast<size_t>(accel_type)]);
			}
			if (density_search.Stopped())
			{
				const happyray::GridDensities best = best_densities[static_cast<size_t>(accel_type)];
				if (accel_type == happyray::AccelType::kUniformGrid)
				{
					std::printf("Final densities [uniform]: density=%.6g\n", best.top_level);
				}
				else
				{
					const char *grid = accel_type == happyray::AccelType::kTwoLevelGrid ? "two-level" : "irregular";
					std::printf("Final densities [%s]: top=%.6g leaf=%.6g\n", grid, best.top_level, best.second_level);
				}
				std::fflush(stdout);
			}
			// Search/rebuild time is not camera-motion time.
			last_ticks = SDL_GetTicks();
		}
		const std::vector<happyray::Vec3f> &resolved = renderer.ResolvedPixels();

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				pixel_buffer[static_cast<size_t>(y) * width + x] =
					ToPackedRgba(resolved[static_cast<size_t>(y) * width + x]);
			}
		}
		if (screenshot_requested)
		{
			const auto		  now		 = std::chrono::system_clock::now();
			const std::time_t timestamp	 = std::chrono::system_clock::to_time_t(now);
			const std::tm	 *local_time = std::localtime(&timestamp);
			char			  date_time[32];
			if (!local_time || std::strftime(date_time, sizeof(date_time), "%Y-%m-%d_%H-%M-%S", local_time) == 0)
			{
				std::fprintf(stderr, "Failed to format screenshot date/time\n");
			}
			else
			{
				const auto microseconds =
					std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
				char path[80];
				std::snprintf(path,
							  sizeof(path),
							  "happyray_%s-%06lld.png",
							  date_time,
							  static_cast<long long>(microseconds));
				WritePngScreenshot(path, width, height, pixel_buffer);
			}
			// Disk I/O time must not become camera-motion time on the next frame.
			last_ticks = SDL_GetTicks();
		}
		const int screenshot_frame_count =
			animation_loaded ? rendered_frame_count : static_cast<int>(renderer.SampleCount());
		if (screenshot_path && screenshot_frame_count >= screenshot_at_frame)
		{
			const std::string path(screenshot_path);
			if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".png") == 0)
			{
				screenshot_failed = !WritePngScreenshot(path, width, height, pixel_buffer);
			}
			else
			{
				WritePpmScreenshot(path, width, height, resolved);
			}
			running = false;
		}
		SDL_UpdateTexture(texture, nullptr, pixel_buffer.data(), width * sizeof(uint32_t));
		SDL_RenderClear(sdl_renderer);
		SDL_RenderCopy(sdl_renderer, texture, nullptr, nullptr);
		SDL_RenderPresent(sdl_renderer);

		if (animation_loaded && !animation.IsPaused() && running)
		{
			animation.Advance();
			const auto build_start = std::chrono::steady_clock::now();
			renderer.UpdateScene(animation.InterpolatedScene());
			build_ms = grid_build_ms_since(build_start);
			renderer.ResetAccumulation();
		}

		char title[256];
		const char	*accel_name = accel_type == happyray::AccelType::kUniformGrid	 ? "ugrid"
								  : accel_type == happyray::AccelType::kTwoLevelGrid ? "2grid"
																					 : "igrid";
		const double frame_seconds =
			std::chrono::duration<double>(std::chrono::steady_clock::now() - frame_start).count();
		const double fps		  = 1.0 / frame_seconds;
		const int	 title_length = std::snprintf(title,
												  sizeof(title),
												  "HappyRay [%s/%s] - %u spp - %.1f fps%s",
												  happyray::RendererBackendName(),
												  accel_name,
												  renderer.SampleCount(),
												  fps,
												  animation_loaded && animation.IsPaused() ? " - paused" : "");
		const happyray::CollisionStats &collision_stats = renderer.LastCollisionStats();
		const bool show_collision_ms = collision_stats.object_count >= 2 && renderer.CollisionDetectionEnabled();
		int		   title_used		 = title_length;
		if (build_ms >= 0 && title_used >= 0 && static_cast<size_t>(title_used) < sizeof(title))
		{
			const int written =
				std::snprintf(title + title_used, sizeof(title) - title_used, " - build: %.2f ms", build_ms);
			title_used = written >= 0 ? title_used + written : -1;
		}
		if (show_collision_ms && title_used >= 0 && static_cast<size_t>(title_used) < sizeof(title))
		{
			std::snprintf(title + title_used, sizeof(title) - title_used, " - coll: %.2f ms", collision_stats.total_ms);
		}
		SDL_SetWindowTitle(window, title);
	}

	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(sdl_renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return screenshot_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
