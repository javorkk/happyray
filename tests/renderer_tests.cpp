// Automated regression test for Renderer (src/render/renderer.h): validates whichever Thrust
// backend this binary was built with (CUDA or TBB -- see CMakeLists.txt's HAPPYRAY_THRUST_BACKEND)
// against a dead-simple, single-threaded, Thrust-free host reference loop that calls
// AOIntegrator::Integrate (src/render/ao_integrator.h) directly. Also validates the uniform grid
// (src/accel/uniform_grid.h), the two-level grid (src/accel/two_level_grid.h), and the
// irregular grid (src/accel/irregular_grid.h) by checking Renderer-with-the-grid-enabled against
// the same brute-force reference: since both trace identical deterministic Halton samples, any
// discrepancy points at the grid's construction/traversal rather than at noise. Each grid is
// checked against two scenes: box_on_floor.obj (forced sparse threshold so
// each grid's degenerate path is exercised) and dense_cube.obj (dense enough that some
// top-level cells get real, non-degenerate refinement -- for the irregular grid, this also
// exercises real merge and expand, not just the flat initial build).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../src/accel/scene_intersect.h"
#include "../src/app/attractor_scene.h"
#include "../src/app/grid_density_search.h"
#include "../src/app/obj_loader.h"
#include "../src/app/scene.h"
#include "../src/core/random.h"
#include "../src/render/camera.h"
#include "../src/render/renderer.h"

namespace
{

	int g_failures = 0;

#define HAPPYRAY_CHECK(cond)                                                                                           \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(cond))                                                                                                   \
		{                                                                                                              \
			std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__);                                   \
			++g_failures;                                                                                              \
		}                                                                                                              \
	} while (0)

	// Reproduces Renderer::ResizeFramebuffer's per-pixel jitter table exactly (same Rng/HashSeed
	// calls), so this reference loop draws the identical Halton samples Renderer does.
	std::vector<happyray::Vec2f> MakePixelJitter(int width, int height)
	{
		std::vector<happyray::Vec2f> jitter(static_cast<size_t>(width) * height);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				happyray::Rng rng(happyray::HashSeed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0u),
								  static_cast<uint64_t>(y) * width + x);
				jitter[static_cast<size_t>(y) * width + x] = happyray::Vec2f(rng.NextFloat(), rng.NextFloat());
			}
		}
		return jitter;
	}

	// Renders `num_frames` accumulated AO samples single-threaded on the host, using brute force
	// only -- this is the project's simplest possible ground truth, with no Thrust/parallelism
	// involved at all.
	template <typename Integrator>
	std::vector<happyray::Vec3f> RenderBruteForceReference(const happyray::SceneView		  &scene,
														   const happyray::Camera			  &camera,
														   const Integrator					  &integrator,
														   int								   width,
														   int								   height,
														   const std::vector<happyray::Vec2f> &pixel_jitter,
														   int								   num_frames,
														   int								   samples_per_frame = 1)
	{
		const happyray::AccelStructures accel; // Default: brute force.
		std::vector<happyray::Vec3f>	accumulated(static_cast<size_t>(width) * height,
													happyray::Vec3f(0.0f, 0.0f, 0.0f));
		for (int frame = 0; frame < num_frames; ++frame)
		{
			const uint32_t sample_offset = static_cast<uint32_t>(frame) * samples_per_frame;
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width; ++x)
				{
					const size_t		idx = static_cast<size_t>(y) * width + x;
					const happyray::Ray ray =
						camera.GenerateRay(static_cast<float>(x), static_cast<float>(y), width, height);
					accumulated[idx] += integrator.Integrate(scene, accel, ray, sample_offset, pixel_jitter[idx]);
				}
			}
		}
		std::vector<happyray::Vec3f> resolved(accumulated.size());
		for (size_t i = 0; i < accumulated.size(); ++i)
		{
			resolved[i] = accumulated[i] / static_cast<float>(num_frames);
		}
		return resolved;
	}

	// Renders `num_frames` accumulated AO samples via `renderer` and checks the result agrees closely
	// with `reference` (the brute-force host loop above), except for up to `max_outlier_fraction` of
	// the pixels.
	template <typename Integrator>
	void CompareRendererToReference(const char						   *label,
									happyray::Renderer				   *renderer,
									const happyray::Camera			   &camera,
									const Integrator				   &integrator,
									int									num_frames,
									const std::vector<happyray::Vec3f> &reference,
									float								max_outlier_fraction = 0.0f)
	{
		renderer->ResetAccumulation();
		for (int frame = 0; frame < num_frames; ++frame)
		{
			renderer->RenderFrame(camera, integrator);
		}
		const std::vector<happyray::Vec3f> &resolved = renderer->ResolvedPixels();

		HAPPYRAY_CHECK(resolved.size() == reference.size());

		float		 max_abs_diff = 0.0f;
		double		 sum_abs_diff = 0.0;
		size_t		 outliers	  = 0;
		const size_t pixel_count  = reference.size();
		for (size_t i = 0; i < pixel_count; ++i)
		{
			const happyray::Vec3f diff = reference[i] - resolved[i];
			const float max_channel	   = std::fmax(std::fabs(diff.x), std::fmax(std::fabs(diff.y), std::fabs(diff.z)));
			max_abs_diff			   = std::fmax(max_channel, max_abs_diff);
			outliers += max_channel >= 0.05f;
			sum_abs_diff += (std::fabs(diff.x) + std::fabs(diff.y) + std::fabs(diff.z)) / 3;
		}
		const float mean_abs_diff = static_cast<float>(sum_abs_diff / pixel_count);

		std::printf("[%s] Renderer (%s) vs host reference: max abs diff = %.4f, mean abs diff = %.5f (%zu pixels)\n",
					label,
					happyray::RendererBackendName(),
					max_abs_diff,
					mean_abs_diff,
					pixel_count);

		// Both traces draw identical Halton samples against identical geometry, so they should agree
		// almost exactly; allow a small tolerance for host/device transcendental function rounding
		// differences (CUDA backend) or summation-order differences (TBB backend).
		HAPPYRAY_CHECK(outliers <= static_cast<size_t>(max_outlier_fraction * pixel_count));
		HAPPYRAY_CHECK(mean_abs_diff < 0.01f);
	}

	void TestRendererLifecycle()
	{
		happyray::Renderer	   renderer(happyray::FlatScene{});
		const happyray::Camera camera;
		happyray::AOIntegrator integrator;
		integrator.background_color			   = happyray::Vec3f(0.2f, 0.4f, 0.6f);
		integrator.samples_per_pixel_per_frame = 4;
		HAPPYRAY_CHECK(renderer.SampleCount() == 0);
		HAPPYRAY_CHECK(renderer.HasAccelerationStructure()); // Brute force needs no build.
		renderer.ResizeFramebuffer(3, 2);
		for (int frame = 0; frame < 3; ++frame)
			renderer.RenderFrame(camera, integrator);
		HAPPYRAY_CHECK(renderer.SampleCount() == 3);
		HAPPYRAY_CHECK(renderer.ResolvedPixels().size() == 6);
		for (const happyray::Vec3f &pixel : renderer.ResolvedPixels())
		{
			HAPPYRAY_CHECK(happyray::Length(pixel - integrator.background_color) < 1e-5f);
		}
		renderer.ResetAccumulation();
		HAPPYRAY_CHECK(renderer.SampleCount() == 0);
		integrator.background_color = happyray::Vec3f(0.8f, 0.6f, 0.4f);
		renderer.RenderFrame(camera, integrator);
		HAPPYRAY_CHECK(renderer.SampleCount() == 1);
		for (const happyray::Vec3f &pixel : renderer.ResolvedPixels())
		{
			HAPPYRAY_CHECK(happyray::Length(pixel - integrator.background_color) < 1e-5f);
		}
		renderer.ResizeFramebuffer(2, 1);
		HAPPYRAY_CHECK(renderer.SampleCount() == 0);
		renderer.RenderFrame(camera, integrator);
		HAPPYRAY_CHECK(renderer.ResolvedPixels().size() == 2);
		for (const happyray::Vec3f &pixel : renderer.ResolvedPixels())
		{
			HAPPYRAY_CHECK(happyray::Length(pixel - integrator.background_color) < 1e-5f);
		}
	}

	void TestProgressiveBlend(const happyray::FlatScene &scene, happyray::Camera camera)
	{
		happyray::Renderer	   progressive(scene), ao_renderer(scene), simple_renderer(scene);
		happyray::AOIntegrator ao;
		ao.samples_per_pixel_per_frame = 3;
		ao.max_distance				   = 1.5f;
		ao.background_color			   = happyray::Vec3f(0.1f, 0.3f, 0.5f);
		happyray::SimpleIntegrator simple;
		simple.background_color = happyray::Vec3f(0.6f, 0.4f, 0.2f);
		simple.light_direction	= camera.Forward() * -1.0f;
		progressive.RenderProgressiveFrame(camera, ao, simple);
		HAPPYRAY_CHECK(progressive.SampleCount() == 0);

		auto check_pixels = [&](const std::vector<happyray::Vec3f> &expected)
		{
			HAPPYRAY_CHECK(progressive.ResolvedPixels().size() == expected.size());
			for (size_t i = 0; i < expected.size(); ++i)
				HAPPYRAY_CHECK(happyray::Length(progressive.ResolvedPixels()[i] - expected[i]) < 1e-5f);
		};
		auto check_sequence = [&](int width, int height)
		{
			progressive.ResizeFramebuffer(width, height);
			ao_renderer.ResizeFramebuffer(width, height);
			simple_renderer.ResizeFramebuffer(width, height);
			simple_renderer.RenderFrame(camera, simple);
			progressive.RenderProgressiveFrame(camera, ao, simple);
			HAPPYRAY_CHECK(progressive.SampleCount() == 1);
			check_pixels(simple_renderer.ResolvedPixels());
			for (uint32_t n = 1; n <= 64; ++n)
			{
				ao_renderer.RenderFrame(camera, ao);
				progressive.RenderProgressiveFrame(camera, ao, simple);
				HAPPYRAY_CHECK(progressive.SampleCount() == n + 1);
				const float					 weight = static_cast<float>(n) / (n + 32.0f);
				std::vector<happyray::Vec3f> expected(static_cast<size_t>(width) * height);
				for (size_t i = 0; i < expected.size(); ++i)
					expected[i] =
						happyray::Lerp(simple_renderer.ResolvedPixels()[i], ao_renderer.ResolvedPixels()[i], weight);
				check_pixels(expected);
			}
		};
		check_sequence(12, 8);
		// Repeated motion resets must never retain AO or a preview from the previous view.
		for (int move = 0; move < 3; ++move)
		{
			camera.MoveRight(0.2f);
			simple.light_direction = camera.Forward() * -1.0f;
			progressive.ResetAccumulation();
			simple_renderer.ResetAccumulation();
			simple_renderer.RenderFrame(camera, simple);
			progressive.RenderProgressiveFrame(camera, ao, simple);
			HAPPYRAY_CHECK(progressive.SampleCount() == 1);
			check_pixels(simple_renderer.ResolvedPixels());
		}
		check_sequence(9, 5);
		progressive.SetAccelType(happyray::AccelType::kUniformGrid);
		progressive.BuildAccelerationStructure();
		progressive.ResetAccumulation();
		progressive.RenderProgressiveFrame(camera, ao, simple);
		check_pixels(simple_renderer.ResolvedPixels());
	}

	void TestAccelerationStructureBuilds(const happyray::FlatScene &scene, const happyray::Camera &camera)
	{
		using namespace happyray;
		Renderer renderer(scene);
		renderer.ResizeFramebuffer(3, 2);
		renderer.BuildAccelerationStructure(); // Brute force needs no allocation.
		for (AccelType type : {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid})
		{
			renderer.SetAccelType(type);
			HAPPYRAY_CHECK(!renderer.HasAccelerationStructure());
			bool rejected = false;
			try
			{
				renderer.RenderFrame(camera, SimpleIntegrator());
			}
			catch (const std::logic_error &)
			{
				rejected = true;
			}
			HAPPYRAY_CHECK(rejected);
			HAPPYRAY_CHECK(renderer.SampleCount() == 0);

			renderer.BuildAccelerationStructure();
			HAPPYRAY_CHECK(renderer.HasAccelerationStructure());
			HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level ==
						   (type == AccelType::kUniformGrid ? 3.0f : 0.12f));
			HAPPYRAY_CHECK(renderer.CurrentGridDensities().second_level ==
						   (type == AccelType::kUniformGrid ? 0.0f : 2.4f));
			renderer.BuildAccelerationStructure({
				{0.25f, 1.5f},
				4,
				0.6f,
				1
			 });
			renderer.UpdateScene(scene);
			HAPPYRAY_CHECK(renderer.HasAccelerationStructure());
			HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == 0.25f);
			HAPPYRAY_CHECK(renderer.CurrentGridDensities().second_level ==
						   (type == AccelType::kUniformGrid ? 0.0f : 1.5f));
			renderer.SetAccelType(AccelType::kBruteForce);
			renderer.SetAccelType(type);
			HAPPYRAY_CHECK(renderer.HasAccelerationStructure()); // Selection retains cached grids.
			HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == 0.25f);
			rejected = false;
			try
			{
				renderer.BuildAccelerationStructure({
					{-1.0f, 1.5f}
				 });
			}
			catch (const std::invalid_argument &)
			{
				rejected = true;
			}
			HAPPYRAY_CHECK(rejected);
			HAPPYRAY_CHECK(renderer.HasAccelerationStructure());
			HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == 0.25f);
		}
	}

	void TestDensitySearchScheduling()
	{
		using namespace happyray;
		GridDensitySearch search;
		const AccelType	  types[] = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid};
		for (AccelType type : types)
		{
			search.Reset(type, 100.0f);
			HAPPYRAY_CHECK(search.CanRebuild(32));	// No waiting for a prior measurement anymore --
													// a SAH cost is available the instant a candidate
													// finishes building.
			HAPPYRAY_CHECK(!search.CanRebuild(31)); // Still needs the view to have settled.
			const GridDensities minimum = search.Candidate();
			float				cost	= 100.0f;
			for (int i = 0; i < 100; ++i)
			{
				HAPPYRAY_CHECK(search.CanRebuild(32));
				const GridDensities candidate = search.Candidate();
				HAPPYRAY_CHECK(candidate.top_level >= minimum.top_level);
				HAPPYRAY_CHECK(candidate.top_level <= minimum.top_level * 16);
				HAPPYRAY_CHECK(candidate.second_level >= minimum.second_level);
				HAPPYRAY_CHECK(candidate.second_level <= minimum.second_level * 16);
				cost *= 0.9f; // Strictly decreasing (but always positive) -- every candidate
							  // improves on the last.
				HAPPYRAY_CHECK(search.Evaluate(cost));
				HAPPYRAY_CHECK(search.CanRebuild(32)); // Next candidate needs no waiting frames.
			}
		}

		search.Reset(AccelType::kUniformGrid, 10.0f);
		for (int i = 0; i < 20; ++i)
		{
			HAPPYRAY_CHECK(search.CanRebuild(32));
			const bool improve = (i == 9); // One improvement resets nine misses.
			HAPPYRAY_CHECK(search.Evaluate(improve ? 5.0f : 20.0f) == improve);
			if (i == 9)
				HAPPYRAY_CHECK(search.Misses() == 0);
		}
		HAPPYRAY_CHECK(search.Attempts() == 20);
		HAPPYRAY_CHECK(search.Stopped());
		HAPPYRAY_CHECK(!search.CanRebuild(1000)); // A stopped search never resumes on its own.

		search.Reset(AccelType::kUniformGrid, 10.0f);
		HAPPYRAY_CHECK(search.Stopped()); // Stopped state persists across Reset for the same type...
		HAPPYRAY_CHECK(!search.CanRebuild(32));
		search.Reset(AccelType::kTwoLevelGrid, 10.0f);
		HAPPYRAY_CHECK(!search.Stopped()); // ...but not for a different one.
		HAPPYRAY_CHECK(search.CanRebuild(32));

		for (AccelType type : {AccelType::kTwoLevelGrid, AccelType::kIrregularGrid})
		{
			search.Reset(type, 10.0f);
			HAPPYRAY_CHECK(!search.Stopped());
			for (int i = 0; i < 10; ++i)
			{
				HAPPYRAY_CHECK(search.CanRebuild(32));
				HAPPYRAY_CHECK(!search.Evaluate(20.0f)); // Strictly worse than the 10.0f baseline.
			}
			HAPPYRAY_CHECK(search.Stopped());
			search.Reset(AccelType::kUniformGrid, 10.0f);
			HAPPYRAY_CHECK(search.Stopped()); // kUniformGrid's own earlier stop, unaffected by `type`.
			search.Reset(type, 10.0f);
			HAPPYRAY_CHECK(search.Stopped());
			HAPPYRAY_CHECK(!search.CanRebuild(1000));
		}
	}

	void TestDensityGridRebuilds(const happyray::FlatScene &scene, const happyray::Camera &camera)
	{
		using namespace happyray;
		Renderer renderer(scene), reference(scene);
		renderer.ResizeFramebuffer(24, 16);
		reference.ResizeFramebuffer(24, 16);
		renderer.SetAccelType(AccelType::kUniformGrid);
		renderer.BuildAccelerationStructure();
		renderer.SetAccelType(AccelType::kTwoLevelGrid);
		renderer.BuildAccelerationStructure({
			{0.0625f, 1.2f},
			4
		 });
		renderer.SetAccelType(AccelType::kIrregularGrid);
		renderer.BuildAccelerationStructure({
			{0.12f, 2.4f},
			4,
			0.6f,
			1
		 });
		AOIntegrator ao;
		ao.max_distance = 1.5f;
		SimpleIntegrator simple;
		const AccelType	 types[] = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid};
		for (AccelType type : types)
		{
			renderer.SetAccelType(type);
			renderer.ResetAccumulation();
			reference.ResetAccumulation();
			renderer.RenderProgressiveFrame(camera, ao, simple);
			reference.RenderProgressiveFrame(camera, ao, simple);
			const GridDensities original = renderer.CurrentGridDensities();
			const GridDensities candidate{original.top_level * 2, original.second_level * 0.5f};
			auto				check_pixels = [&]
			{
				HAPPYRAY_CHECK(renderer.SampleCount() == reference.SampleCount());
				for (size_t i = 0; i < reference.ResolvedPixels().size(); ++i)
					HAPPYRAY_CHECK(Length(renderer.ResolvedPixels()[i] - reference.ResolvedPixels()[i]) < 1e-3f);
			};
			for (bool keep : {false, true})
			{
				const uint32_t samples = renderer.SampleCount();
				renderer.RebuildGridDensities(candidate);
				HAPPYRAY_CHECK(renderer.SampleCount() == samples); // No hidden trial renders.
				check_pixels();
				HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == candidate.top_level);
				renderer.RenderProgressiveFrame(camera, ao, simple);
				reference.RenderProgressiveFrame(camera, ao, simple);
				check_pixels();
				if (!keep)
					renderer.RebuildGridDensities(original);
				check_pixels();
				const GridDensities expected = keep ? candidate : original;
				HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == expected.top_level);
				HAPPYRAY_CHECK(renderer.CurrentGridDensities().second_level == expected.second_level);
				renderer.RenderProgressiveFrame(camera, ao, simple);
				reference.RenderProgressiveFrame(camera, ao, simple);
				check_pixels();
			}
			renderer.RebuildGridDensities(original);
			renderer.SetAccelType(AccelType::kBruteForce);
			renderer.SetAccelType(type);
			renderer.RebuildGridDensities(candidate);
			HAPPYRAY_CHECK(renderer.CurrentGridDensities().top_level == candidate.top_level);
		}
	}

	void TestAnimatedSceneUpdate(const happyray::FlatScene &scene, const happyray::Camera &camera)
	{
		using namespace happyray;
		Renderer grid_renderer(scene), reference_renderer(scene);
		grid_renderer.SetAccelType(AccelType::kUniformGrid);
		FlatScene updated_scene = scene;
		for (Vec3f &position : updated_scene.positions)
		{
			position.z -= 0.35f;
		}
		grid_renderer.UpdateScene(updated_scene);
		reference_renderer.UpdateScene(updated_scene);
		HAPPYRAY_CHECK(grid_renderer.HasAccelerationStructure());

		grid_renderer.ResizeFramebuffer(12, 8);
		reference_renderer.ResizeFramebuffer(12, 8);
		AOIntegrator integrator;
		integrator.max_distance = 1.5f;
		grid_renderer.RenderFrame(camera, integrator);
		reference_renderer.RenderFrame(camera, integrator);
		HAPPYRAY_CHECK(grid_renderer.ResolvedPixels().size() == reference_renderer.ResolvedPixels().size());
		for (size_t i = 0; i < reference_renderer.ResolvedPixels().size(); ++i)
		{
			HAPPYRAY_CHECK(Length(grid_renderer.ResolvedPixels()[i] - reference_renderer.ResolvedPixels()[i]) < 1e-3f);
		}
	}

	// Appends an n x n tessellated parallelogram (origin + [0,1]^2 * (u, v)) tagged `object_id`.
	void AddTessellatedQuad(happyray::FlatScene	  *scene,
							const happyray::Vec3f &origin,
							const happyray::Vec3f &u,
							const happyray::Vec3f &v,
							int					   n,
							uint32_t			   object_id)
	{
		using namespace happyray;
		const Vec3f	   normal = Normalize(Cross(u, v));
		const uint32_t base	  = static_cast<uint32_t>(scene->positions.size());
		for (int j = 0; j <= n; ++j)
		{
			for (int i = 0; i <= n; ++i)
			{
				scene->positions.push_back(origin + u * (static_cast<float>(i) / n) + v * (static_cast<float>(j) / n));
				scene->normals.push_back(normal);
			}
		}
		for (int j = 0; j < n; ++j)
		{
			for (int i = 0; i < n; ++i)
			{
				const uint32_t a = base + j * (n + 1) + i;
				const uint32_t b = a + 1, c = a + (n + 1), d = c + 1;
				scene->indices.insert(scene->indices.end(), {a, b, d, a, d, c});
				scene->material_ids.insert(scene->material_ids.end(), {0u, 0u});
				scene->object_ids.insert(scene->object_ids.end(), {object_id, object_id});
			}
		}
	}

	// All-pairs host oracle: per primitive, the smallest other object id `collide(a, b)` reports.
	template <typename CollideFn>
	std::vector<uint32_t> CollisionOracle(const happyray::FlatScene &scene, CollideFn &&collide)
	{
		using namespace happyray;
		const SceneView		  view = scene.View();
		std::vector<uint32_t> expected(view.primitive_count, kNoCollision);
		for (uint32_t a = 0; a < view.primitive_count; ++a)
		{
			for (uint32_t b = a + 1; b < view.primitive_count; ++b)
			{
				const uint32_t object_a = scene.object_ids[a], object_b = scene.object_ids[b];
				if (object_a != object_b && collide(FetchTriangle(view, a), FetchTriangle(view, b)))
				{
					expected[a] = std::min(expected[a], object_b);
					expected[b] = std::min(expected[b], object_a);
				}
			}
		}
		return expected;
	}

	uint32_t CountCollisions(const std::vector<uint32_t> &ids)
	{
		return static_cast<uint32_t>(
			std::count_if(ids.begin(), ids.end(), [](uint32_t id) { return id != happyray::kNoCollision; }));
	}

	// Primitives sharing a leaf cell collide when their boxes overlap: every truly intersecting pair
	// is found, and only bbox-overlapping pairs are (all of them with AABB cell assignment).
	bool CollisionsWithinBounds(const std::vector<uint32_t> &ids,
								const std::vector<uint32_t> &bbox_expected,
								const std::vector<uint32_t> &exact_expected)
	{
		if (ids.size() != bbox_expected.size())
			return false;
		for (size_t i = 0; i < ids.size(); ++i)
		{
			if (ids[i] < bbox_expected[i] || ids[i] > exact_expected[i])
				return false;
		}
		return true;
	}

	// Every grid's leaf-cell collision detection must agree with all-pairs host oracles, and
	// colliding primitives must render with the red collided material.
	void TestCollisionDetection()
	{
		using namespace happyray;
		FlatScene scene;
		scene.material_diffuse_colors = {Vec3f(1.0f, 1.0f, 1.0f)};
		// Object 0: floor plus a wall piercing it (a self-intersection, which must be ignored).
		AddTessellatedQuad(&scene, Vec3f(-1, 0, -1), Vec3f(2, 0, 0), Vec3f(0, 0, 2), 16, 0);
		AddTessellatedQuad(&scene, Vec3f(0.73f, -0.3f, -0.2f), Vec3f(0, 0, 0.4f), Vec3f(0, 0.6f, 0), 4, 0);
		// Object 1: a wall crossing the floor. Object 2: a ceiling touching nothing.
		AddTessellatedQuad(&scene, Vec3f(-0.5f, -0.5f, 0.03f), Vec3f(1, 0, 0), Vec3f(0, 1, 0), 16, 1);
		AddTessellatedQuad(&scene, Vec3f(-1, 0.8f, -1), Vec3f(0, 0, 2), Vec3f(2, 0, 0), 8, 2);

		const SceneView				view = scene.View();
		const std::vector<uint32_t> bbox_expected =
			CollisionOracle(scene,
							[](const Triangle &a, const Triangle &b)
							{ return Overlaps(BBoxExtractor<Triangle>::Get(a), BBoxExtractor<Triangle>::Get(b)); });
		const std::vector<uint32_t> exact_expected =
			CollisionOracle(scene, [](const Triangle &a, const Triangle &b) { return IntersectTriangles(a, b); });
		bool ceiling_untouched = true;
		for (uint32_t prim = 0; prim < view.primitive_count; ++prim)
		{
			if (scene.object_ids[prim] == 2 && bbox_expected[prim] != kNoCollision)
				ceiling_untouched = false;
		}
		HAPPYRAY_CHECK(CountCollisions(exact_expected) > 0);
		HAPPYRAY_CHECK(CountCollisions(bbox_expected) >= CountCollisions(exact_expected));
		HAPPYRAY_CHECK(ceiling_untouched);

		Renderer renderer(scene);
		HAPPYRAY_CHECK(renderer.CollisionIds() == std::vector<uint32_t>(view.primitive_count, kNoCollision));
		const AccelType		  grids[] = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid};
		const CellOverlapMode modes[] = {CellOverlapMode::kAabb, CellOverlapMode::kExact, CellOverlapMode::kHybrid};
		for (CellOverlapMode mode : modes)
		{
			renderer.SetCellOverlapMode(mode);
			for (AccelType type : grids)
			{
				renderer.SetAccelType(type);
				renderer.BuildAccelerationStructure();
				renderer.DetectCollisions();
				const std::vector<uint32_t> ids = renderer.CollisionIds();
				HAPPYRAY_CHECK(CollisionsWithinBounds(ids, bbox_expected, exact_expected));
				if (mode == CellOverlapMode::kAabb)
					HAPPYRAY_CHECK(ids == bbox_expected);
				HAPPYRAY_CHECK(renderer.LastCollisionStats().object_count == 3);
				HAPPYRAY_CHECK(renderer.LastCollisionStats().candidate_cell_count > 0);
				HAPPYRAY_CHECK(renderer.LastCollisionStats().colliding_primitive_count == CountCollisions(ids));
			}
		}
		renderer.SetCellOverlapMode(DefaultCellOverlapMode());
		// Refined second levels and small irregular cells exercise more leaves than the defaults.
		renderer.SetAccelType(AccelType::kTwoLevelGrid);
		renderer.BuildAccelerationStructure({
			{0.5f, 8.0f},
			2
		 });
		HAPPYRAY_CHECK(CollisionsWithinBounds(renderer.CollisionIds(), bbox_expected, exact_expected));
		renderer.SetAccelType(AccelType::kIrregularGrid);
		renderer.BuildAccelerationStructure({
			{0.5f, 8.0f},
			2
		 });
		const std::vector<uint32_t> expected	   = renderer.CollisionIds();
		const uint32_t				expected_count = CountCollisions(expected);
		HAPPYRAY_CHECK(CollisionsWithinBounds(expected, bbox_expected, exact_expected));

		// Disabled detection (a paused animation) keeps the last results through rebuilds.
		renderer.SetCollisionDetectionEnabled(false);
		HAPPYRAY_CHECK(!renderer.CollisionDetectionEnabled());
		const float last_collision_ms = renderer.LastCollisionStats().total_ms;
		renderer.SetAccelType(AccelType::kUniformGrid);
		renderer.BuildAccelerationStructure({
			{1.0f, 0}
		 });
		renderer.SetAccelType(AccelType::kTwoLevelGrid);
		HAPPYRAY_CHECK(renderer.CollisionIds() == expected);
		HAPPYRAY_CHECK(renderer.LastCollisionStats().total_ms == last_collision_ms);
		HAPPYRAY_CHECK(renderer.LastCollisionStats().colliding_primitive_count == expected_count);
		renderer.SetCollisionDetectionEnabled(true);
		renderer.SetAccelType(AccelType::kIrregularGrid);
		HAPPYRAY_CHECK(CollisionsWithinBounds(renderer.CollisionIds(), bbox_expected, exact_expected));

		// Toggling off clears the results; an explicit detection runs even while disabled.
		renderer.SetCollisionDetectionEnabled(false);
		renderer.ClearCollisions();
		HAPPYRAY_CHECK(renderer.CollisionIds() == std::vector<uint32_t>(view.primitive_count, kNoCollision));
		HAPPYRAY_CHECK(renderer.LastCollisionStats().colliding_primitive_count == 0);
		renderer.DetectCollisions();
		HAPPYRAY_CHECK(CollisionsWithinBounds(renderer.CollisionIds(), bbox_expected, exact_expected));
		HAPPYRAY_CHECK(renderer.LastCollisionStats().colliding_primitive_count ==
					   CountCollisions(renderer.CollisionIds()));
		renderer.SetCollisionDetectionEnabled(true);

		Camera camera;
		camera.position	 = Vec3f(0.0f, 0.3f, 2.0f);
		camera.yaw_deg	 = -90.0f;
		camera.pitch_deg = -10.0f;
		renderer.ResizeFramebuffer(64, 48);
		renderer.RenderFrame(camera, SimpleIntegrator());
		int red_pixels = 0;
		for (const Vec3f &pixel : renderer.ResolvedPixels())
		{
			if (pixel.x > 0.1f && pixel.y == 0.0f && pixel.z == 0.0f)
				++red_pixels;
		}
		HAPPYRAY_CHECK(red_pixels > 0);

		// A single-object scene has nothing to collide with.
		FlatScene single_object = scene;
		single_object.object_ids.clear();
		renderer.UpdateScene(single_object);
		HAPPYRAY_CHECK(renderer.LastCollisionStats().object_count == 1);
		HAPPYRAY_CHECK(renderer.LastCollisionStats().colliding_primitive_count == 0);
		HAPPYRAY_CHECK(renderer.CollisionIds() == std::vector<uint32_t>(view.primitive_count, kNoCollision));
	}

	// Host copy of `scene` with Renderer's screen-space point/curve radii for `camera`.
	happyray::FlatScene
	WithScreenSpaceRadii(happyray::FlatScene scene, const happyray::Camera &camera, int height, float pixels)
	{
		using namespace happyray;
		const float world_per_pixel =
			pixels * 2.0f * std::tan(0.5f * camera.vertical_fov_deg * (3.14159265358979323846f / 180.0f)) / height;
		auto radius = [&](const Vec3f &a, const Vec3f &b)
		{ return world_per_pixel * std::fmin(Length(a - camera.position), Length(b - camera.position)); };
		for (Point &point : scene.points)
			point.radius = radius(point.vtx[0], point.vtx[0]);
		for (BezierCurve &curve : scene.curves)
			curve.radius = radius(curve.vtx[0], curve.vtx[3]);
		return scene;
	}

	// Every grid in every cell-overlap mode must match the host brute-force reference.
	void TestSceneAcrossOverlapModes(const char					  *label,
									 const happyray::FlatScene	  &flat_scene,
									 const happyray::Camera		  &camera,
									 const happyray::AOIntegrator &integrator)
	{
		using namespace happyray;
		constexpr int kWidth = 48, kHeight = 27, kFrames = 2;
		const bool	  view_dependent = flat_scene.primitive_type == PrimitiveType::kPoint ||
									flat_scene.primitive_type == PrimitiveType::kBezierCurve;
		const FlatScene reference_scene =
			view_dependent ? WithScreenSpaceRadii(flat_scene, camera, kHeight, 2.0f) : flat_scene;
		const std::vector<Vec2f> jitter	   = MakePixelJitter(kWidth, kHeight);
		const std::vector<Vec3f> reference = RenderBruteForceReference(reference_scene.View(),
																	   camera,
																	   integrator,
																	   kWidth,
																	   kHeight,
																	   jitter,
																	   kFrames,
																	   integrator.samples_per_pixel_per_frame);
		int						 covered   = 0;
		for (const Vec3f &pixel : reference)
			covered += Length(pixel - integrator.background_color) > 1e-3f;
		HAPPYRAY_CHECK(covered > kWidth * kHeight / 20);

		Renderer renderer(flat_scene);
		renderer.ResizeFramebuffer(kWidth, kHeight);
		HAPPYRAY_CHECK(renderer.UpdateViewDependentGeometry(camera) == view_dependent);
		const CellOverlapMode modes[] = {CellOverlapMode::kAabb, CellOverlapMode::kExact, CellOverlapMode::kHybrid};
		const AccelType		  types[] = {AccelType::kUniformGrid, AccelType::kTwoLevelGrid, AccelType::kIrregularGrid};
		// Grazing silhouette hits may flip between host and device rounding.
		renderer.SetAccelType(AccelType::kBruteForce);
		const std::string brute_label = std::string(label) + " brute force";
		CompareRendererToReference(brute_label.c_str(), &renderer, camera, integrator, kFrames, reference, 0.005f);
		const std::vector<Vec3f> device_reference = renderer.ResolvedPixels();
		for (CellOverlapMode mode : modes)
		{
			renderer.SetCellOverlapMode(mode);
			for (AccelType type : types)
			{
				renderer.SetAccelType(type);
				renderer.BuildAccelerationStructure();
				const std::string case_label = std::string(label) + " " + CellOverlapModeName(mode) + " " +
											   (type == AccelType::kUniformGrid	   ? "uniform"
												: type == AccelType::kTwoLevelGrid ? "two-level"
																				   : "irregular");
				CompareRendererToReference(case_label.c_str(),
										   &renderer,
										   camera,
										   integrator,
										   kFrames,
										   device_reference);
			}
		}
	}

	void TestPrimitiveScenes(const happyray::FlatScene &triangle_scene, const happyray::Camera &triangle_camera)
	{
		using namespace happyray;
		AOIntegrator integrator;
		integrator.samples_per_pixel_per_frame = 2;
		integrator.max_distance				   = 1.5f;
		TestSceneAcrossOverlapModes("triangles", triangle_scene, triangle_camera, integrator);

		const PrimitiveType types[] = {PrimitiveType::kPoint, PrimitiveType::kSphere, PrimitiveType::kBezierCurve};
		for (PrimitiveType type : types)
		{
			const Scene scene		= MakeAttractorScene(type, 1500);
			Camera		camera		= MakeDefaultCamera(scene);
			camera.vertical_fov_deg = 30.0f;
			integrator.max_distance = DefaultMaxDistance(scene);
			TestSceneAcrossOverlapModes(PrimitiveTypeName(type), FlattenScene(scene), camera, integrator);
		}

		// Larger screen-space radii cover more pixels.
		const Scene	 scene	= MakeAttractorScene(PrimitiveType::kBezierCurve, 1500);
		const Camera camera = MakeDefaultCamera(scene);
		Renderer	 renderer(FlattenScene(scene));
		renderer.ResizeFramebuffer(48, 27);
		renderer.SetAccelType(AccelType::kTwoLevelGrid);
		int coverage[2] = {0, 0};
		for (int i = 0; i < 2; ++i)
		{
			renderer.SetScreenSpaceRadius(i == 0 ? 1.0f : 4.0f);
			HAPPYRAY_CHECK(renderer.UpdateViewDependentGeometry(camera));
			renderer.ResetAccumulation();
			renderer.RenderFrame(camera, SimpleIntegrator());
			for (const Vec3f &pixel : renderer.ResolvedPixels())
				coverage[i] += Length(pixel - SimpleIntegrator().background_color) > 1e-3f;
		}
		HAPPYRAY_CHECK(coverage[1] > coverage[0]);
	}

} // namespace

int main(int argc, char **argv)
{
	const std::string assets_dir = (argc > 1) ? argv[1] : "assets";

	TestRendererLifecycle();
	TestDensitySearchScheduling();
	TestCollisionDetection();
	happyray::Scene scene;
	HAPPYRAY_CHECK(happyray::LoadObj(assets_dir + "/box_on_floor.obj", &scene));
	const happyray::FlatScene flat_scene = happyray::FlattenScene(scene);
	const happyray::SceneView view		 = flat_scene.View();

	constexpr int kWidth	 = 96;
	constexpr int kHeight	 = 54;
	constexpr int kNumFrames = 8;

	happyray::Camera camera;
	camera.position	 = happyray::Vec3f(0.0f, 1.8f, 3.5f);
	camera.yaw_deg	 = -90.0f;
	camera.pitch_deg = -25.0f;

	TestProgressiveBlend(flat_scene, camera);
	TestAccelerationStructureBuilds(flat_scene, camera);
	TestDensityGridRebuilds(flat_scene, camera);
	TestAnimatedSceneUpdate(flat_scene, camera);
	TestPrimitiveScenes(flat_scene, camera);

	happyray::AOIntegrator integrator;
	integrator.samples_per_pixel_per_frame = 4;
	integrator.max_distance				   = 1.5f;

	const std::vector<happyray::Vec2f> pixel_jitter = MakePixelJitter(kWidth, kHeight);
	const std::vector<happyray::Vec3f> reference	= RenderBruteForceReference(view,
																				camera,
																				integrator,
																				kWidth,
																				kHeight,
																				pixel_jitter,
																				kNumFrames,
																				integrator.samples_per_pixel_per_frame);

	happyray::Renderer renderer(flat_scene);
	renderer.ResizeFramebuffer(kWidth, kHeight);

	renderer.SetAccelType(happyray::AccelType::kBruteForce);
	CompareRendererToReference("brute force", &renderer, camera, integrator, kNumFrames, reference);

	renderer.SetAccelType(happyray::AccelType::kUniformGrid);
	renderer.BuildAccelerationStructure();
	CompareRendererToReference("uniform grid", &renderer, camera, integrator, kNumFrames, reference);

	// Scene-loading subdivision increases triangle counts; force the sparse path explicitly.
	renderer.SetAccelType(happyray::AccelType::kTwoLevelGrid);
	renderer.BuildAccelerationStructure({
		{0.0625f, 1.2f},
		std::numeric_limits<uint32_t>::max()
	});
	CompareRendererToReference("two-level grid (degenerate)", &renderer, camera, integrator, kNumFrames, reference);

	renderer.SetAccelType(happyray::AccelType::kIrregularGrid);
	renderer.BuildAccelerationStructure({
		{0.12f, 2.4f},
		std::numeric_limits<uint32_t>::max()
	});
	CompareRendererToReference("irregular grid (degenerate)", &renderer, camera, integrator, kNumFrames, reference);

	happyray::SimpleIntegrator simple_integrator;
	simple_integrator.light_direction = camera.Forward() * -1.0f;
	const auto simple_reference =
		RenderBruteForceReference(view, camera, simple_integrator, kWidth, kHeight, pixel_jitter, 1);
	const happyray::AccelType simple_accels[] = {happyray::AccelType::kBruteForce,
												 happyray::AccelType::kUniformGrid,
												 happyray::AccelType::kTwoLevelGrid,
												 happyray::AccelType::kIrregularGrid};
	for (happyray::AccelType type : simple_accels)
	{
		renderer.SetAccelType(type);
		CompareRendererToReference("simple integrator", &renderer, camera, simple_integrator, 2, simple_reference);
		HAPPYRAY_CHECK(renderer.SampleCount() == 2);
	}

	// The subdivided dense cube exercises real second-level refinement at default densities.
	happyray::Scene dense_scene;
	HAPPYRAY_CHECK(happyray::LoadObj(assets_dir + "/dense_cube.obj", &dense_scene));
	const happyray::FlatScene dense_flat_scene = happyray::FlattenScene(dense_scene);
	const happyray::SceneView dense_view	   = dense_flat_scene.View();

	happyray::Camera dense_camera;
	dense_camera.position  = happyray::Vec3f(1.6f, 1.2f, 2.2f);
	dense_camera.yaw_deg   = -145.0f;
	dense_camera.pitch_deg = -20.0f;

	happyray::AOIntegrator dense_settings;
	dense_settings.samples_per_pixel_per_frame = 4;
	dense_settings.max_distance				   = 0.6f;

	const std::vector<happyray::Vec3f> dense_reference =
		RenderBruteForceReference(dense_view,
								  dense_camera,
								  dense_settings,
								  kWidth,
								  kHeight,
								  pixel_jitter,
								  kNumFrames,
								  dense_settings.samples_per_pixel_per_frame);

	happyray::Renderer dense_renderer(dense_flat_scene);
	dense_renderer.ResizeFramebuffer(kWidth, kHeight);
	dense_renderer.SetAccelType(happyray::AccelType::kTwoLevelGrid);
	dense_renderer.BuildAccelerationStructure();
	CompareRendererToReference("two-level grid (refined)",
							   &dense_renderer,
							   dense_camera,
							   dense_settings,
							   kNumFrames,
							   dense_reference);

	// Same scene, now checking the irregular grid: its per-top-level-cell local grids are dense
	// enough here to exercise real merge (collapsing empty/uniform regions) and expand
	// (growing cells into subset-compatible neighbors), not just the degenerate case above.
	dense_renderer.SetAccelType(happyray::AccelType::kIrregularGrid);
	dense_renderer.BuildAccelerationStructure();
	CompareRendererToReference("irregular grid (merge+expand)",
							   &dense_renderer,
							   dense_camera,
							   dense_settings,
							   kNumFrames,
							   dense_reference);

	if (g_failures == 0)
	{
		std::printf("All renderer tests passed.\n");
		return EXIT_SUCCESS;
	}
	std::fprintf(stderr, "%d renderer test check(s) failed.\n", g_failures);
	return EXIT_FAILURE;
}
