// Thrust implementation of the render engine (see renderer.h). Evaluates the integrator
// for each camera ray with a plain Thrust functor invoked via thrust::for_each -- so
// this file is parallelized entirely through Thrust algorithms, with no CUDA-specific launch
// syntax (<<<...>>>, blockIdx/threadIdx, cudaGetLastError/cudaDeviceSynchronize). That is what
// lets the exact same source compile either under Thrust's CUDA backend (nvcc, runs on the GPU)
// or its TBB backend (plain g++, runs in parallel on the CPU) -- see CMakeLists.txt's
// HAPPYRAY_THRUST_BACKEND option, which picks the backend by choosing the compiler/language this
// file is built with (and, for TBB, defining THRUST_HOST_SYSTEM/THRUST_DEVICE_SYSTEM below).
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_TBB
#endif

#include "renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "../accel/irregular_grid.h"
#include "../accel/two_level_grid.h"
#include "../accel/uniform_grid.h"
#include "../core/random.h"
#include "collision_detector.h"

namespace happyray
{
	namespace
	{

		// Camera-ray generation and integrator dispatch, independent of the shading algorithm.
		template <typename Integrator, typename P> struct RenderPixelFunctor
		{
			SceneView		scene;
			AccelStructures accel;
			Camera			camera;
			Integrator		integrator;
			const Vec2f	   *pixel_jitter;
			Vec3f		   *accumulated;
			uint32_t		sample_offset;
			int				width;
			int				height;

			HAPPYRAY_HD void operator()(uint32_t pixel_index) const
			{
				const int x			  = static_cast<int>(pixel_index % width);
				const int y			  = static_cast<int>(pixel_index / width);
				const Ray primary_ray = camera.GenerateRay(static_cast<float>(x), static_cast<float>(y), width, height);
				const Vec3f sample	  = integrator.template Integrate<P>(scene,
																		 accel,
																		 primary_ray,
																		 sample_offset,
																		 pixel_jitter[pixel_index]);
				accumulated[pixel_index] += sample;
			}
		};

		// Resolves the frame average, optionally fading it over a separate preview.
		struct ResolveFunctor
		{
			float		 inv_sample_count;
			const Vec3f *accumulated;
			Vec3f		*resolved;
			const Vec3f *preview;
			float		 ao_weight;

			HAPPYRAY_HD void operator()(uint32_t pixel) const
			{
				const Vec3f average = accumulated[pixel] * inv_sample_count;
				resolved[pixel]		= preview ? Lerp(preview[pixel], average, ao_weight) : average;
			}
		};

		// Sets the radius to `world_per_pixel` times the distance from `eye` to the nearest endpoint.
		template <typename P> struct ScreenSpaceRadiusFunctor
		{
			Vec3f eye;
			float world_per_pixel;

			HAPPYRAY_HD P operator()(P primitive) const
			{
				const float distance =
					fminf(Length(primitive.vtx[0] - eye), Length(primitive.vtx[P::kNumVertices - 1] - eye));
				primitive.radius = world_per_pixel * distance;
				return primitive;
			}
		};

	} // namespace

	struct Renderer::Impl
	{
		PrimitiveType					   primitive_type = PrimitiveType::kTriangle;
		thrust::device_vector<Vec3f>	   positions;
		thrust::device_vector<Vec3f>	   normals;
		thrust::device_vector<uint32_t>	   indices;
		thrust::device_vector<Point>	   points;
		thrust::device_vector<Sphere>	   spheres;
		thrust::device_vector<BezierCurve> curves;
		// Scene-provided points/curves; `points`/`curves` hold the view-dependent copies.
		thrust::device_vector<Point>	   scene_points;
		thrust::device_vector<BezierCurve> scene_curves;
		float							   screen_space_radius = 2.0f;
		bool							   has_view_camera	   = false;
		Camera							   view_camera;
		CellOverlapMode					   overlap_mode = DefaultCellOverlapMode();
		thrust::device_vector<uint32_t>	   material_ids;
		thrust::device_vector<Vec3f>	   material_diffuse_colors; // Ends with the collided material.
		thrust::device_vector<uint32_t>	   object_ids;
		thrust::device_vector<uint32_t>	   collision_ids;
		uint32_t						   object_count = 1;
		CollisionDetector				   collision_detector;
		CollisionStats					   collision_stats;
		bool							   collision_detection_enabled = true;

		thrust::device_vector<Vec3f> accumulated;
		thrust::device_vector<Vec2f> pixel_jitter;
		thrust::device_vector<Vec3f> resolved_device;
		thrust::device_vector<Vec3f> preview;
		std::vector<Vec3f>			 resolved_host;

		std::unique_ptr<UniformGrid>	 uniform_grid;
		std::unique_ptr<TwoLevelGrid>	 two_level_grid;
		std::unique_ptr<IrregularGrid>	 irregular_grid;
		AccelType						 accel_type = AccelType::kBruteForce;
		std::array<GridBuildSettings, 4> grid_settings;

		int		 width		  = 0;
		int		 height		  = 0;
		uint32_t sample_count = 0;

		void UploadScene(const FlatScene &scene);
		void ApplyViewDependentRadii();

		// Runs collision detection over `grid`'s leaf cells, refreshing collision_ids.
		template <typename Grid> void RunCollisionDetection(const Grid &grid)
		{
			collision_stats = collision_detector.ComputeCollisions(View(),
																   object_count,
																   grid,
																   thrust::raw_pointer_cast(collision_ids.data()));
		}

		// Per-build detection, skipped while disabled.
		template <typename Grid> void ComputeCollisions(const Grid &grid)
		{
			if (collision_detection_enabled)
				RunCollisionDetection(grid);
		}

		template <typename Integrator>
		void RenderFrame(const Camera	  &camera,
						 const Integrator &integrator,
						 uint32_t		   sample_offset,
						 uint32_t		   accumulated_frames,
						 const Vec3f	  *preview_pixels = nullptr,
						 float			   ao_weight	  = 1.0f);

		SceneView View() const
		{
			SceneView view;
			view.positions				 = thrust::raw_pointer_cast(positions.data());
			view.normals				 = thrust::raw_pointer_cast(normals.data());
			view.indices				 = thrust::raw_pointer_cast(indices.data());
			view.material_ids			 = thrust::raw_pointer_cast(material_ids.data());
			view.material_diffuse_colors = thrust::raw_pointer_cast(material_diffuse_colors.data());
			view.object_ids				 = thrust::raw_pointer_cast(object_ids.data());
			view.collision_ids			 = thrust::raw_pointer_cast(collision_ids.data());
			view.collided_material_id	 = static_cast<uint32_t>(material_diffuse_colors.size() - 1);
			view.primitive_count		 = static_cast<uint32_t>(material_ids.size());
			view.primitive_type			 = primitive_type;
			view.points					 = thrust::raw_pointer_cast(points.data());
			view.spheres				 = thrust::raw_pointer_cast(spheres.data());
			view.curves					 = thrust::raw_pointer_cast(curves.data());
			return view;
		}

		AccelStructures CurrentAccel() const
		{
			AccelStructures accel;
			accel.type = accel_type;
			if (accel_type == AccelType::kUniformGrid && uniform_grid)
			{
				accel.uniform_grid = uniform_grid->View();
			}
			else if (accel_type == AccelType::kTwoLevelGrid && two_level_grid)
			{
				accel.two_level_grid = two_level_grid->View();
			}
			else if (accel_type == AccelType::kIrregularGrid && irregular_grid)
			{
				accel.irregular_grid = irregular_grid->View();
			}
			else if (accel_type != AccelType::kBruteForce)
			{
				throw std::logic_error("Selected acceleration structure has not been built");
			}
			return accel;
		}
	};

	// Red diffuse color that replaces the material of primitives colliding with another object.
	const Vec3f kCollidedDiffuseColor(1.0f, 0.0f, 0.0f);

	void Renderer::Impl::UploadScene(const FlatScene &scene)
	{
		const size_t primitive_count = scene.PrimitiveCount();
		if (scene.material_ids.size() != primitive_count)
			throw std::invalid_argument("Scenes need one material id per primitive");
		primitive_type = scene.primitive_type;
		positions.assign(scene.positions.begin(), scene.positions.end());
		normals.assign(scene.normals.begin(), scene.normals.end());
		indices.assign(scene.indices.begin(), scene.indices.end());
		spheres.assign(scene.spheres.begin(), scene.spheres.end());
		scene_points.assign(scene.points.begin(), scene.points.end());
		scene_curves.assign(scene.curves.begin(), scene.curves.end());
		ApplyViewDependentRadii();
		material_ids.assign(scene.material_ids.begin(), scene.material_ids.end());
		std::vector<Vec3f> colors = scene.material_diffuse_colors;
		colors.push_back(kCollidedDiffuseColor);
		material_diffuse_colors.assign(colors.begin(), colors.end());

		object_count = 1;
		if (scene.object_ids.size() == primitive_count)
		{
			object_ids.assign(scene.object_ids.begin(), scene.object_ids.end());
			for (uint32_t object_id : scene.object_ids)
			{
				object_count = std::max(object_count, object_id + 1);
			}
		}
		else
		{
			object_ids.assign(primitive_count, 0u);
		}
		collision_ids.assign(primitive_count, kNoCollision);
		collision_stats				 = CollisionStats{};
		collision_stats.object_count = object_count;
	}

	void Renderer::Impl::ApplyViewDependentRadii()
	{
		points = scene_points;
		curves = scene_curves;
		if (!has_view_camera || screen_space_radius <= 0.0f || height <= 0)
			return;
		const float tan_half_fov = std::tan(0.5f * view_camera.vertical_fov_deg * (3.14159265358979323846f / 180.0f));
		const float world_per_pixel = screen_space_radius * 2.0f * tan_half_fov / static_cast<float>(height);
		thrust::transform(scene_points.begin(),
						  scene_points.end(),
						  points.begin(),
						  ScreenSpaceRadiusFunctor<Point>{view_camera.position, world_per_pixel});
		thrust::transform(scene_curves.begin(),
						  scene_curves.end(),
						  curves.begin(),
						  ScreenSpaceRadiusFunctor<BezierCurve>{view_camera.position, world_per_pixel});
	}

	Renderer::Renderer(const FlatScene &scene) : impl_(std::make_unique<Impl>())
	{
		impl_->UploadScene(scene);
	}

	Renderer::~Renderer() = default;

	void Renderer::ResizeFramebuffer(int width, int height)
	{
		impl_->width			 = width;
		impl_->height			 = height;
		const size_t pixel_count = static_cast<size_t>(width) * height;

		impl_->accumulated.assign(pixel_count, Vec3f(0.0f, 0.0f, 0.0f));
		impl_->resolved_device.resize(pixel_count);
		impl_->resolved_host.resize(pixel_count);
		impl_->sample_count = 0;

		// A stable per-pixel random offset that decorrelates the otherwise pixel-independent Halton
		// sequence (see random.h); generated once per resolution on the host, then uploaded.
		std::vector<Vec2f> jitter(pixel_count);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				Rng rng(HashSeed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0u),
						static_cast<uint64_t>(y) * width + x);
				jitter[static_cast<size_t>(y) * width + x] = Vec2f(rng.NextFloat(), rng.NextFloat());
			}
		}
		impl_->pixel_jitter.assign(jitter.begin(), jitter.end());
	}

	void Renderer::UpdateScene(const FlatScene &scene)
	{
		impl_->UploadScene(scene);
		RebuildGrids();
	}

	void Renderer::SetScreenSpaceRadius(float pixels)
	{
		impl_->screen_space_radius = pixels;
	}

	float Renderer::ScreenSpaceRadius() const
	{
		return impl_->screen_space_radius;
	}

	bool Renderer::UpdateViewDependentGeometry(const Camera &camera)
	{
		impl_->view_camera	   = camera;
		impl_->has_view_camera = true;
		if (impl_->primitive_type != PrimitiveType::kPoint && impl_->primitive_type != PrimitiveType::kBezierCurve)
			return false;
		impl_->ApplyViewDependentRadii();
		RebuildGrids();
		return true;
	}

	void Renderer::SetCellOverlapMode(CellOverlapMode mode)
	{
		if (impl_->overlap_mode == mode)
			return;
		impl_->overlap_mode = mode;
		RebuildGrids();
	}

	CellOverlapMode Renderer::GetCellOverlapMode() const
	{
		return impl_->overlap_mode;
	}

	void Renderer::RebuildGrids()
	{
		impl_->uniform_grid.reset();
		impl_->two_level_grid.reset();
		impl_->irregular_grid.reset();
		BuildAccelerationStructure(impl_->grid_settings[static_cast<size_t>(impl_->accel_type)]);
	}

	void Renderer::ResetAccumulation()
	{
		thrust::fill(impl_->accumulated.begin(), impl_->accumulated.end(), Vec3f(0.0f, 0.0f, 0.0f));
		impl_->sample_count = 0;
	}

	uint32_t Renderer::SampleCount() const
	{
		return impl_->sample_count;
	}

	void Renderer::BuildAccelerationStructure(GridBuildSettings settings)
	{
		if (impl_->accel_type == AccelType::kBruteForce)
			return;
		auto &densities = settings.densities;
		if (!std::isfinite(densities.top_level) || densities.top_level < 0 ||
			(impl_->accel_type != AccelType::kUniformGrid &&
			 (!std::isfinite(densities.second_level) || densities.second_level < 0)))
			throw std::invalid_argument("Grid densities must be finite and nonnegative (zero selects defaults)");
		if (densities.top_level == 0)
			densities.top_level = impl_->accel_type == AccelType::kUniformGrid ? UniformGrid::kDefaultDensity
																			   : TwoLevelGrid::kDefaultTopLevelDensity;
		if (densities.second_level == 0 && impl_->accel_type != AccelType::kUniformGrid)
			densities.second_level = 2.4f;
		switch (impl_->accel_type)
		{
		case AccelType::kUniformGrid:
			densities.second_level = 0;
			impl_->uniform_grid.reset();
			impl_->uniform_grid =
				std::make_unique<UniformGrid>(impl_->View(), densities.top_level, impl_->overlap_mode);
			impl_->ComputeCollisions(*impl_->uniform_grid);
			break;
		case AccelType::kTwoLevelGrid:
			impl_->two_level_grid.reset();
			impl_->two_level_grid = std::make_unique<TwoLevelGrid>(impl_->View(),
																   densities.top_level,
																   densities.second_level,
																   settings.dense_cell_threshold,
																   impl_->overlap_mode);
			impl_->ComputeCollisions(*impl_->two_level_grid);
			break;
		case AccelType::kIrregularGrid:
			impl_->irregular_grid.reset();
			impl_->irregular_grid = std::make_unique<IrregularGrid>(impl_->View(),
																	densities.top_level,
																	densities.second_level,
																	settings.dense_cell_threshold,
																	settings.merge_alpha,
																	settings.expand_iterations,
																	impl_->overlap_mode);
			impl_->ComputeCollisions(*impl_->irregular_grid);
			break;
		default:
			break;
		}
		impl_->grid_settings[static_cast<size_t>(impl_->accel_type)] = settings;
	}

	bool Renderer::HasAccelerationStructure() const
	{
		switch (impl_->accel_type)
		{
		case AccelType::kUniformGrid:
			return impl_->uniform_grid != nullptr;
		case AccelType::kTwoLevelGrid:
			return impl_->two_level_grid != nullptr;
		case AccelType::kIrregularGrid:
			return impl_->irregular_grid != nullptr;
		default:
			return true;
		}
	}

	void Renderer::SetAccelType(AccelType type)
	{
		if (type < AccelType::kBruteForce || type > AccelType::kIrregularGrid)
			throw std::invalid_argument("Unknown acceleration structure type");
		impl_->accel_type = type;
		if (impl_->collision_detection_enabled)
			DetectCollisions();
	}

	AccelType Renderer::GetAccelType() const
	{
		return impl_->accel_type;
	}

	GridDensities Renderer::CurrentGridDensities() const
	{
		if (impl_->accel_type == AccelType::kBruteForce)
			throw std::logic_error("Density tuning requires an active grid");
		return impl_->grid_settings[static_cast<size_t>(impl_->accel_type)].densities;
	}

	void Renderer::RebuildGridDensities(GridDensities candidate)
	{
		if (!std::isfinite(candidate.top_level) || candidate.top_level <= 0 ||
			(impl_->accel_type != AccelType::kUniformGrid &&
			 (!std::isfinite(candidate.second_level) || candidate.second_level <= 0)))
			throw std::invalid_argument("Grid densities must be finite and positive");

		if (impl_->accel_type == AccelType::kBruteForce)
			throw std::logic_error("Density tuning requires an active grid");
		if (!HasAccelerationStructure())
			throw std::logic_error("Selected acceleration structure has not been built");
		auto settings	   = impl_->grid_settings[static_cast<size_t>(impl_->accel_type)];
		settings.densities = candidate;
		BuildAccelerationStructure(settings);
	}

	float Renderer::ComputeCurrentGridSahCost(const BBox &scene_bounds, const SahCostConstants &costs) const
	{
		switch (impl_->accel_type)
		{
		case AccelType::kUniformGrid:
			if (!impl_->uniform_grid)
				throw std::logic_error("Uniform grid has not been built");
			return impl_->uniform_grid->ComputeSahCost(scene_bounds, costs);
		case AccelType::kTwoLevelGrid:
			if (!impl_->two_level_grid)
				throw std::logic_error("Two-level grid has not been built");
			return impl_->two_level_grid->ComputeSahCost(scene_bounds, costs);
		case AccelType::kIrregularGrid:
			if (!impl_->irregular_grid)
				throw std::logic_error("Irregular grid has not been built");
			return impl_->irregular_grid->ComputeSahCost(scene_bounds, costs);
		default:
			throw std::logic_error("SAH cost computation requires an active grid");
		}
	}

	template <typename Integrator>
	void Renderer::Impl::RenderFrame(const Camera	  &camera,
									 const Integrator &integrator,
									 uint32_t		   sample_offset,
									 uint32_t		   accumulated_frames,
									 const Vec3f	  *preview_pixels,
									 float			   ao_weight)
	{
		if (width <= 0 || height <= 0)
		{
			return;
		}

		const uint32_t pixel_count = static_cast<uint32_t>(width) * height;

		DispatchPrimitive(primitive_type,
						  [&](auto tag)
						  {
							  using P = typename decltype(tag)::type;
							  const RenderPixelFunctor<Integrator, P> render{
								  View(),
								  CurrentAccel(),
								  camera,
								  integrator,
								  thrust::raw_pointer_cast(pixel_jitter.data()),
								  thrust::raw_pointer_cast(accumulated.data()),
								  sample_offset,
								  width,
								  height};
							  thrust::for_each(thrust::counting_iterator<uint32_t>(0),
											   thrust::counting_iterator<uint32_t>(pixel_count),
											   render);
						  });
		++sample_count;

		const ResolveFunctor resolve{1.0f / static_cast<float>(accumulated_frames),
									 thrust::raw_pointer_cast(accumulated.data()),
									 thrust::raw_pointer_cast(resolved_device.data()),
									 preview_pixels,
									 ao_weight};
		thrust::for_each(thrust::counting_iterator<uint32_t>(0),
						 thrust::counting_iterator<uint32_t>(pixel_count),
						 resolve);
		thrust::copy(resolved_device.begin(), resolved_device.end(), resolved_host.begin());
	}

	void Renderer::RenderFrame(const Camera &camera, const AOIntegrator &integrator)
	{
		impl_->RenderFrame(camera,
						   integrator,
						   impl_->sample_count * integrator.samples_per_pixel_per_frame,
						   impl_->sample_count + 1);
	}

	void Renderer::RenderFrame(const Camera &camera, const SimpleIntegrator &integrator)
	{
		impl_->RenderFrame(camera, integrator, 0, impl_->sample_count + 1);
	}

	void Renderer::RenderProgressiveFrame(const Camera &camera, const AOIntegrator &ao, const SimpleIntegrator &simple)
	{
		if (impl_->width <= 0 || impl_->height <= 0)
			return;
		if (impl_->sample_count == 0)
		{
			RenderFrame(camera, simple);
			impl_->preview = impl_->resolved_device;
			thrust::fill(impl_->accumulated.begin(), impl_->accumulated.end(), Vec3f());
			return;
		}
		const uint32_t ao_frames = impl_->sample_count;
		const float	   ao_weight = static_cast<float>(ao_frames) / (static_cast<float>(ao_frames) + 32.0f);
		impl_->RenderFrame(camera,
						   ao,
						   (ao_frames - 1) * ao.samples_per_pixel_per_frame,
						   ao_frames,
						   thrust::raw_pointer_cast(impl_->preview.data()),
						   ao_weight);
	}

	const std::vector<Vec3f> &Renderer::ResolvedPixels() const
	{
		return impl_->resolved_host;
	}

	std::vector<uint32_t> Renderer::CollisionIds() const
	{
		std::vector<uint32_t> ids(impl_->collision_ids.size());
		thrust::copy(impl_->collision_ids.begin(), impl_->collision_ids.end(), ids.begin());
		return ids;
	}

	const CollisionStats &Renderer::LastCollisionStats() const
	{
		return impl_->collision_stats;
	}

	void Renderer::SetCollisionDetectionEnabled(bool enabled)
	{
		impl_->collision_detection_enabled = enabled;
	}

	bool Renderer::CollisionDetectionEnabled() const
	{
		return impl_->collision_detection_enabled;
	}

	void Renderer::DetectCollisions()
	{
		switch (impl_->accel_type)
		{
		case AccelType::kUniformGrid:
			if (impl_->uniform_grid)
				impl_->RunCollisionDetection(*impl_->uniform_grid);
			break;
		case AccelType::kTwoLevelGrid:
			if (impl_->two_level_grid)
				impl_->RunCollisionDetection(*impl_->two_level_grid);
			break;
		case AccelType::kIrregularGrid:
			if (impl_->irregular_grid)
				impl_->RunCollisionDetection(*impl_->irregular_grid);
			break;
		default:
			break;
		}
	}

	void Renderer::ClearCollisions()
	{
		thrust::fill(impl_->collision_ids.begin(), impl_->collision_ids.end(), kNoCollision);
		impl_->collision_stats				= CollisionStats{};
		impl_->collision_stats.object_count = impl_->object_count;
	}

	const char *RendererBackendName()
	{
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
		return "TBB";
#else
		return "CUDA";
#endif
	}

} // namespace happyray
