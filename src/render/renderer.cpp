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

#include <cmath>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "random.h"
#include "irregular_grid.h"
#include "two_level_grid.h"
#include "uniform_grid.h"

namespace happyray {
namespace {

// Camera-ray generation and integrator dispatch, independent of the shading algorithm.
template <typename Integrator>
struct RenderPixelFunctor {
    SceneView scene;
    AccelStructures accel;
    Camera camera;
    Integrator integrator;
    const Vec2* pixel_jitter;
    Vec3* accumulated;
    uint32_t sample_offset;
    int width;
    int height;

    HAPPYRAY_HD void operator()(uint32_t pixel_index) const {
        const int x = static_cast<int>(pixel_index % width);
        const int y = static_cast<int>(pixel_index / width);
        const Ray primary_ray =
            camera.GenerateRay(static_cast<float>(x), static_cast<float>(y), width, height);
        const Vec3 sample = integrator.Integrate(scene, accel, primary_ray, sample_offset,
                                                pixel_jitter[pixel_index]);
        accumulated[pixel_index] += sample;
    }
};

// Resolves the frame average, optionally fading it over a separate preview.
struct ResolveFunctor {
    float inv_sample_count;
    const Vec3* accumulated;
    Vec3* resolved;
    const Vec3* preview;
    float ao_weight;

    HAPPYRAY_HD void operator()(uint32_t pixel) const {
        const Vec3 average = accumulated[pixel] * inv_sample_count;
        resolved[pixel] = preview ? Lerp(preview[pixel], average, ao_weight) : average;
    }
};

}  // namespace

struct Renderer::Impl {
    thrust::device_vector<Vec3> positions;
    thrust::device_vector<Vec3> normals;
    thrust::device_vector<uint32_t> indices;
    thrust::device_vector<uint32_t> material_ids;
    thrust::device_vector<Vec3> material_diffuse_colors;

    thrust::device_vector<Vec3> accumulated;
    thrust::device_vector<Vec2> pixel_jitter;
    thrust::device_vector<Vec3> resolved_device;
    thrust::device_vector<Vec3> preview;
    std::vector<Vec3> resolved_host;

    std::unique_ptr<UniformGrid> uniform_grid;
    std::unique_ptr<TwoLevelGrid> two_level_grid;
    std::unique_ptr<IrregularGrid> irregular_grid;
    AccelType accel_type = AccelType::kBruteForce;
    GridDensities uniform_densities;
    GridDensities two_level_densities;
    GridDensities irregular_densities;
    uint32_t two_level_threshold = 16;
    uint32_t irregular_threshold = 16;
    float irregular_merge_alpha = 0.7f;
    int irregular_expand_iterations = 2;

    int width = 0;
    int height = 0;
    uint32_t sample_count = 0;

    template <typename Integrator>
    void RenderFrame(const Camera& camera, const Integrator& integrator, uint32_t sample_offset,
                     uint32_t accumulated_frames, const Vec3* preview_pixels = nullptr,
                     float ao_weight = 1.0f);

    SceneView View() const {
        SceneView view;
        view.positions = thrust::raw_pointer_cast(positions.data());
        view.normals = thrust::raw_pointer_cast(normals.data());
        view.indices = thrust::raw_pointer_cast(indices.data());
        view.material_ids = thrust::raw_pointer_cast(material_ids.data());
        view.material_diffuse_colors = thrust::raw_pointer_cast(material_diffuse_colors.data());
        view.triangle_count = static_cast<uint32_t>(indices.size() / 3);
        return view;
    }

    AccelStructures CurrentAccel() const {
        AccelStructures accel;
        accel.type = accel_type;
        if (accel_type == AccelType::kUniformGrid && uniform_grid) {
            accel.uniform_grid = uniform_grid->View();
        } else if (accel_type == AccelType::kTwoLevelGrid && two_level_grid) {
            accel.two_level_grid = two_level_grid->View();
        } else if (accel_type == AccelType::kIrregularGrid && irregular_grid) {
            accel.irregular_grid = irregular_grid->View();
        }
        return accel;
    }
};

Renderer::Renderer(const FlatScene& scene) : impl_(std::make_unique<Impl>()) {
    impl_->positions.assign(scene.positions.begin(), scene.positions.end());
    impl_->normals.assign(scene.normals.begin(), scene.normals.end());
    impl_->indices.assign(scene.indices.begin(), scene.indices.end());
    impl_->material_ids.assign(scene.material_ids.begin(), scene.material_ids.end());
    impl_->material_diffuse_colors.assign(scene.material_diffuse_colors.begin(),
                                          scene.material_diffuse_colors.end());
}

Renderer::~Renderer() = default;

void Renderer::ResizeFramebuffer(int width, int height) {
    impl_->width = width;
    impl_->height = height;
    const size_t pixel_count = static_cast<size_t>(width) * height;

    impl_->accumulated.assign(pixel_count, Vec3(0.0f, 0.0f, 0.0f));
    impl_->resolved_device.resize(pixel_count);
    impl_->resolved_host.resize(pixel_count);
    impl_->sample_count = 0;

    // A stable per-pixel random offset that decorrelates the otherwise pixel-independent Halton
    // sequence (see random.h); generated once per resolution on the host, then uploaded.
    std::vector<Vec2> jitter(pixel_count);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Rng rng(HashSeed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), 0u),
                   static_cast<uint64_t>(y) * width + x);
            jitter[static_cast<size_t>(y) * width + x] = Vec2(rng.NextFloat(), rng.NextFloat());
        }
    }
    impl_->pixel_jitter.assign(jitter.begin(), jitter.end());
}

void Renderer::ResetAccumulation() {
    thrust::fill(impl_->accumulated.begin(), impl_->accumulated.end(), Vec3(0.0f, 0.0f, 0.0f));
    impl_->sample_count = 0;
}

uint32_t Renderer::SampleCount() const { return impl_->sample_count; }

void Renderer::BuildUniformGrid(float target_triangles_per_cell) {
    impl_->uniform_grid = std::make_unique<UniformGrid>(impl_->View(), target_triangles_per_cell);
    impl_->uniform_densities = {target_triangles_per_cell, 0};
}

bool Renderer::HasUniformGrid() const { return impl_->uniform_grid != nullptr; }

void Renderer::BuildTwoLevelGrid(float top_level_density, float second_level_density,
                              uint32_t dense_cell_threshold) {
    impl_->two_level_grid = std::make_unique<TwoLevelGrid>(
        impl_->View(), top_level_density, second_level_density, dense_cell_threshold);
    impl_->two_level_densities = {top_level_density, second_level_density};
    impl_->two_level_threshold = dense_cell_threshold;
}

bool Renderer::HasTwoLevelGrid() const { return impl_->two_level_grid != nullptr; }

void Renderer::BuildIrregularGrid(float top_level_density, float second_level_density,
                               uint32_t dense_cell_threshold, float merge_alpha,
                               int expand_iterations) {
    impl_->irregular_grid =
        std::make_unique<IrregularGrid>(impl_->View(), top_level_density, second_level_density,
                                        dense_cell_threshold, merge_alpha, expand_iterations);
    impl_->irregular_densities = {top_level_density, second_level_density};
    impl_->irregular_threshold = dense_cell_threshold;
    impl_->irregular_merge_alpha = merge_alpha;
    impl_->irregular_expand_iterations = expand_iterations;
}

bool Renderer::HasIrregularGrid() const { return impl_->irregular_grid != nullptr; }

void Renderer::SetAccelType(AccelType type) { impl_->accel_type = type; }

AccelType Renderer::GetAccelType() const { return impl_->accel_type; }

GridDensities Renderer::CurrentGridDensities() const {
    switch (impl_->accel_type) {
        case AccelType::kUniformGrid: return impl_->uniform_densities;
        case AccelType::kTwoLevelGrid: return impl_->two_level_densities;
        case AccelType::kIrregularGrid: return impl_->irregular_densities;
        default: throw std::logic_error("Density tuning requires an active grid");
    }
}

void Renderer::RebuildGridDensities(GridDensities candidate) {
    if (!std::isfinite(candidate.top_level) || candidate.top_level <= 0 ||
        (impl_->accel_type != AccelType::kUniformGrid &&
         (!std::isfinite(candidate.second_level) || candidate.second_level <= 0)))
        throw std::invalid_argument("Grid densities must be finite and positive");

    switch (impl_->accel_type) {
        case AccelType::kUniformGrid:
            if (!impl_->uniform_grid) throw std::logic_error("Uniform grid has not been built");
            impl_->uniform_grid.reset();
            BuildUniformGrid(candidate.top_level);
            break;
        case AccelType::kTwoLevelGrid:
            if (!impl_->two_level_grid) throw std::logic_error("Two-level grid has not been built");
            impl_->two_level_grid.reset();
            BuildTwoLevelGrid(candidate.top_level, candidate.second_level, impl_->two_level_threshold);
            break;
        case AccelType::kIrregularGrid:
            if (!impl_->irregular_grid) throw std::logic_error("Irregular grid has not been built");
            impl_->irregular_grid.reset();
            BuildIrregularGrid(candidate.top_level, candidate.second_level, impl_->irregular_threshold,
                               impl_->irregular_merge_alpha, impl_->irregular_expand_iterations);
            break;
        default: throw std::logic_error("Density tuning requires an active grid");
    }
}

float Renderer::ComputeCurrentGridSahCost(const BBox& scene_bounds,
                                         const SahCostConstants& costs) const {
    switch (impl_->accel_type) {
        case AccelType::kUniformGrid:
            if (!impl_->uniform_grid) throw std::logic_error("Uniform grid has not been built");
            return impl_->uniform_grid->ComputeSahCost(scene_bounds, costs);
        case AccelType::kTwoLevelGrid:
            if (!impl_->two_level_grid) throw std::logic_error("Two-level grid has not been built");
            return impl_->two_level_grid->ComputeSahCost(scene_bounds, costs);
        case AccelType::kIrregularGrid:
            if (!impl_->irregular_grid) throw std::logic_error("Irregular grid has not been built");
            return impl_->irregular_grid->ComputeSahCost(scene_bounds, costs);
        default: throw std::logic_error("SAH cost computation requires an active grid");
    }
}

template <typename Integrator>
void Renderer::Impl::RenderFrame(const Camera& camera, const Integrator& integrator,
                                 uint32_t sample_offset, uint32_t accumulated_frames,
                                 const Vec3* preview_pixels, float ao_weight) {
    if (width <= 0 || height <= 0) {
        return;
    }

    const uint32_t pixel_count = static_cast<uint32_t>(width) * height;

    RenderPixelFunctor<Integrator> render{View(),
                             CurrentAccel(),
                             camera,
                             integrator,
                             thrust::raw_pointer_cast(pixel_jitter.data()),
                             thrust::raw_pointer_cast(accumulated.data()),
                             sample_offset,
                             width,
                             height};
    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(pixel_count), render);
    ++sample_count;

    const ResolveFunctor resolve{1.0f / static_cast<float>(accumulated_frames),
                                  thrust::raw_pointer_cast(accumulated.data()),
                                  thrust::raw_pointer_cast(resolved_device.data()),
                                  preview_pixels, ao_weight};
    thrust::for_each(thrust::counting_iterator<uint32_t>(0),
                     thrust::counting_iterator<uint32_t>(pixel_count), resolve);
    thrust::copy(resolved_device.begin(), resolved_device.end(), resolved_host.begin());
}

void Renderer::RenderFrame(const Camera& camera, const AOIntegrator& integrator) {
    impl_->RenderFrame(camera, integrator,
                       impl_->sample_count * integrator.samples_per_pixel_per_frame,
                       impl_->sample_count + 1);
}

void Renderer::RenderFrame(const Camera& camera, const SimpleIntegrator& integrator) {
    impl_->RenderFrame(camera, integrator, 0, impl_->sample_count + 1);
}

void Renderer::RenderProgressiveFrame(const Camera& camera, const AOIntegrator& ao,
                                       const SimpleIntegrator& simple) {
    if (impl_->width <= 0 || impl_->height <= 0) return;
    if (impl_->sample_count == 0) {
        RenderFrame(camera, simple);
        impl_->preview = impl_->resolved_device;
        thrust::fill(impl_->accumulated.begin(), impl_->accumulated.end(), Vec3());
        return;
    }
    const uint32_t ao_frames = impl_->sample_count;
    const float ao_weight = static_cast<float>(ao_frames) /
                            (static_cast<float>(ao_frames) + 32.0f);
    impl_->RenderFrame(camera, ao, (ao_frames - 1) * ao.samples_per_pixel_per_frame,
                       ao_frames, thrust::raw_pointer_cast(impl_->preview.data()), ao_weight);
}

const std::vector<Vec3>& Renderer::ResolvedPixels() const { return impl_->resolved_host; }

const char* RendererBackendName() {
#if HAPPYRAY_THRUST_BACKEND_IS_TBB
    return "TBB";
#else
    return "CUDA";
#endif
}

}  // namespace happyray
