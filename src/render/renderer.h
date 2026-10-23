// Thrust-backed render engine, exposed through a plain-C++ interface
// so it can be used from ordinary .cpp translation units (like
// src/app/main.cpp) without that caller needing to know or care which Thrust
// backend is active. There is exactly one implementation (renderer.cpp),
// parallelized entirely via Thrust algorithms/functors (no hand-written CUDA
// kernels) -- which hardware it actually runs on (GPU via CUDA, or CPU via TBB)
// is a build-time choice (CMake's HAPPYRAY_THRUST_BACKEND option), not a runtime
// one, so a given binary only ever has one backend compiled in.
#pragma once

#include <memory>
#include <vector>

#include "ao_integrator.h"
#include "camera.h"
#include "sah_cost.h"
#include "scene.h"
#include "scene_intersect.h"
#include "simple_integrator.h"
#include "vec.h"

namespace happyray {

struct GridDensities {
  float top_level = 0;
  float second_level = 0; // Unused for a uniform grid.
};

class Renderer {
public:
  // Uploads `scene`'s geometry once; the Renderer keeps its own copy, so
  // `scene` doesn't need to outlive this call. ("Uploads" means an actual
  // host->device copy for the CUDA backend, or just an ordinary host-side copy
  // for the TBB backend.)
  explicit Renderer(const FlatScene &scene);
  ~Renderer();
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  // (Re)allocates the accumulation buffer and generates a fresh per-pixel
  // Cranley-Patterson jitter table (see random.h) for the new resolution.
  void ResizeFramebuffer(int width, int height);

  // Restarts progressive accumulation (e.g. after the camera moves) without
  // regenerating the pixel jitter table.
  void ResetAccumulation();

  uint32_t SampleCount() const;

  // Builds a uniform grid over the already-uploaded geometry (reusing it, not
  // re-uploading it). Select it with SetAccelType(). Safe to call more than
  // once (rebuilds in place).
  void BuildUniformGrid(float target_triangles_per_cell = 3.0f);
  bool HasUniformGrid() const;

  // Builds a two-level grid over the already-uploaded geometry; see
  // TwoLevelGrid (src/accel/two_level_grid.h) for the parameter defaults'
  // meaning.
  void BuildTwoLevelGrid(float top_level_density = 0.12f,
                         float second_level_density = 2.4f,
                         uint32_t dense_cell_threshold = 16);
  bool HasTwoLevelGrid() const;

  // Builds an irregular grid over the already-uploaded geometry; see
  // IrregularGrid (src/accel/irregular_grid.h) for the parameter defaults'
  // meaning.
  void BuildIrregularGrid(float top_level_density = 0.12f,
                          float second_level_density = 2.4f,
                          uint32_t dense_cell_threshold = 16,
                          float merge_alpha = 0.7f, int expand_iterations = 2);
  bool HasIrregularGrid() const;

  // Selects which acceleration structure RenderFrame traces against;
  // kUniformGrid/ kTwoLevelGrid/kIrregularGrid require the matching Build*() to
  // have been called first. Does not reset accumulation itself -- callers that
  // want a clean image after switching should call ResetAccumulation() too.
  void SetAccelType(AccelType type);
  AccelType GetAccelType() const;

  GridDensities CurrentGridDensities() const;
  // Releases and rebuilds the active grid without rendering or resetting
  // accumulation.
  void RebuildGridDensities(GridDensities densities);

  // Computes the currently active grid's structural SAH cost (see
  // src/accel/sah_cost.h) from its own already-built cell layout and the
  // supplied scene bounds/cost constants -- requires an active, already-built
  // uniform/two-level/irregular grid (not brute force). Used by the grid
  // density search (src/app/grid_density_search.h) to prefer one candidate
  // density over another without casting any rays.
  float ComputeCurrentGridSahCost(const BBox &scene_bounds,
                                  const SahCostConstants &costs) const;

  // Generates camera rays and evaluates the supplied integrator for each pixel,
  // accumulates one frame, resolves the frame average, and copies it to
  // ResolvedPixels(). Reset accumulation after changing the camera or
  // integrator configuration.
  void RenderFrame(const Camera &camera, const AOIntegrator &integrator);
  void RenderFrame(const Camera &camera, const SimpleIntegrator &integrator);

  // First frame after reset/resize is simple-only; later frames average AO and
  // blend it over that preview with weight n/(n+32), where n is the number of
  // AO frames. Reset before switching between progressive and single-integrator
  // rendering.
  void RenderProgressiveFrame(const Camera &camera, const AOIntegrator &ao,
                              const SimpleIntegrator &simple);

  const std::vector<Vec3> &ResolvedPixels() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A short, human-readable label for whichever Thrust backend this binary was
// built with (e.g. "CUDA" or "TBB"), for display purposes (the interactive
// viewer's window title).
const char *RendererBackendName();

} // namespace happyray
