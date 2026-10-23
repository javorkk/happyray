# HappyRay v2.0

A small ray tracer implemented in [Thrust](https://developer.nvidia.com/thrust).
Currently supports fly-throughs in static .obj scenes in an interactive SDL2 window.
Implements:

  * [A Parallel Algorithm for Construction of Uniform Grids](https://javor.me/data/gpu_grid_construction.pdf)
  * [Two-Level Grids for Ray Tracing on GPUs](https://javor.me/data/two_level_grids.pdf)
  * [GPU Ray Tracing using Irregular Grids](https://javor.me/data/irregular_grid.pdf)

## Building

Dependencies (CMake 3.25+, SDL2, libpng 1.6+, a GCC host compiler, CUDA and optionally TBB).
CUDA toolkit required for thrust headers even if building for CPU.

```sh
cd HappyRay
mkdir -p build && cd build
cmake -S .. -B .                              # -DHAPPYRAY_THRUST_BACKEND=CUDA|TBB
cmake --build . -j"$(nproc)"
ctest --output-on-failure
```

Switch between GPU and CPU backends using `HAPPYRAY_THRUST_BACKEND=CUDA|TBB` (default `CUDA`).
Use `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to specify GPU architectyre (default `89`). 

## X forwarding using Docker on WSL2

In WSL2:
```sh
touch ~/.Xauthority
xauth add ${HOST}:0 . $(xxd -l 16 -p /dev/urandom)
xauth list
```
The result should look like:
```sh
IGOR-II/unix:0  MIT-MAGIC-COOKIE-1  9e6238034158c72ba4b08487c0ff0cd0
```
Then in the running docker image:
```sh
xauth add ${DISPLAY} MIT-MAGIC-COOKIE-1  9e6238034158c72ba4b08487c0ff0cd0
```

## Running the viewer

```sh
./build/src/app/happyray assets/box_on_floor.obj [width] [height]
```

The window starts at an explicit `[width] [height]` if given; otherwise at the scene's sibling
`<name>.camera.cfg`'s `resX`/`resY` if it has one (see below); otherwise at a 512x512 default.
The window is resizable at runtime (drag an edge/corner) -- the framebuffer and on-screen texture
are reallocated at the new size and accumulation restarts.

Controls: `W/A/S/D` to move, `Q`/`Z` to rise/descend, left-click inside the window to grab the
mouse for looking around (click again to release it), `[`/`]` to halve/double the movement step
size (initially 5% of the scene's bounding-box diagonal), `Space` to reset the camera to its
starting viewpoint, `C` to print the current camera's parameters and window resolution
(happyray's camera.cfg text format, including its `resX`/`resY` fields -- see `scene.h`'s
`LoadCameraConfig`) to the console, and `F1` to print this control list.
`HAPPYRAY_LOOK_SENSITIVITY=<value>` seeds a starting value instead of the default `0.1`, and each
wheel adjustment prints the exact value to set via that variable to keep it across runs; pitch
is additionally scaled by a fixed `0.256` factor relative to yaw (not user-adjustable, to keep
the control scheme simple), so side-to-side turning is the dominant/faster rotation, matching
how most FPS-style cameras feel -- yaw sweeps the full 360 degrees while pitch only ever covers
a bounded +-89 degrees, so it doesn't need as much per-pixel precision; `1`, `2`, and `3` select the
uniform grid, the two-level grid (the default), and the irregular grid acceleration structures
(happyray has no equivalent, since it doesn't support switching acceleration structures at
runtime). `T` saves the current rendered image (including accumulated AO) as
`happyray_<date>_<time>.png` (for example, `happyray_2026-10-22_17-30-45-123456.png`) in the
current working directory. Holding the key does not generate repeated screenshots.
Setting the environment variables `HAPPYRAY_SCREENSHOT_PATH=/path/to/out.ppm` and
`HAPPYRAY_SCREENSHOT_FRAME=<n>` dumps a PPM screenshot after `<n>` accumulated frames and exits
(use a `.png` path instead to save PNG), and
`HAPPYRAY_ACCEL_TYPE=uniform_grid`, `HAPPYRAY_ACCEL_TYPE=two_level_grid`, or
`HAPPYRAY_ACCEL_TYPE=irregular_grid` selects that structure from startup instead of the default, for
headless verification. A scene can also have a sibling `<name>.camera.cfg` giving it a fixed
starting viewpoint (and, via `resX`/`resY`, starting resolution) instead of the generic
bounding-box-framed default camera at the 512x512 default resolution -- see
the bundled scene camera configuration files.

There is no integrator environment-variable switch. After `n` stationary AO frames, the display
is `(1 - w) * simple + w * average_AO`, with `w = n / (n + 32)`: the first AO frame has only
1/33 weight, 32 AO frames give 50%, and the image tends to pure AO as rendering progresses.
AO samples are averaged equally in a separate accumulator; the preview never contaminates the
AO estimate. The screenshot frame count includes the initial simple-only frame.

After 32 still frames, the viewer searches the active grid's densities in
`[0.25x, 4x]` its original defaults (both top/leaf densities for nested grids).
A candidate is preferred over the current best purely by
structural SAH (surface-area-heuristic) cost (see `src/accel/sah_cost.h`) -- computed directly
from the candidate's own just-built cell layout, with no rays cast at all -- using
`traversal`/`intersection` cost constants measured once, at startup, by timing each
acceleration structure's own real traversal code and one shared ray/triangle intersection test
against small synthetic test cases (`happyray::MeasureSahCalibration()`).
This adaptive search is disabled for scenes with more than 1,000,000 primitives; those scenes
keep the grid's default densities.

## Running the benchmark

```sh
./build/src/app/happyray_benchmark [scene.obj ...] [--width W] [--height H] [--frames N]
                               [--warmup N] [--samples N] [--assets-dir DIR]
```

Headless (no SDL/window dependency): times `Renderer::RenderFrame` for every acceleration
structure (brute force, uniform grid, two-level grid) against each given scene (or, with no
scene arguments, `assets/box_on_floor.obj` and `assets/dense_cube.obj`), after a few untimed
warmup frames, and prints a table of mean/min/max frame time and estimated AO-ray throughput
(Mrays/s). Like the viewer, it only ever exercises whichever Thrust backend it was built with —
build both backends and run each one's `happyray_benchmark` to compare them.

### Acceleration performance tests (CUDA)

The default CUDA CTest suite tests the uniform, two-level, and irregular grids on
Fairy Forest, Conference, and Ogre. Each test constructs each grid **once** and traces
**one batch of 1,048,576 random closest-hit rays**, with no warmups or repeated measurements.
There is no string2shape comparison or primary-ray pass. All OBJ loading, including the
renderer, bisects each triangle's longest edge until no edge exceeds 0.1 times the
original scene bounding-box diagonal. Subdivision preserves winding and material IDs,
linearly interpolates normals/UVs, and happens before upload and timed grid construction.
The limit is fixed for the whole scene, not recomputed per triangle or child.
Performance tests verify every resulting edge and print the limit and maximum edge.

```sh
cmake --build build -j4
ctest --test-dir build --output-on-failure
ctest --test-dir build -L performance -V   # Show tables even for passing tests.
```

Each table contains build time, random-ray traversal time/throughput, persistent memory
footprint, and structural SAH cost (see `src/accel/sah_cost.h`). The tests report whether
build time, random throughput, memory, and SAH cost increase/decrease (SAH cost is expected to
*decrease*) in the order uniform < two-level < irregular. These hardware/scene-dependent
comparisons are **informational**, not pass/fail requirements. Only correctness and execution
errors fail tests. Grid names include their default densities (top/leaf for nested grids).
Successful ordering and closest-hit checks are silent; only `NOT OBSERVED` ordering
comparisons and correctness failures are printed.
`--output-on-failure` follows standard CTest behavior: passing tables are
suppressed; use `-V` to see them or read `Testing/Temporary/LastTest.log` in the build.

Each grid uses its own default construction parameters, including densities and sparse-cell
thresholds, exactly as a default-constructed grid in the renderer.

Random origins are uniform within the scene AABB, directions are isotropic, and PCG
seed/sequence are 20261021/17. Every grid receives the identical batch. Full-batch
results must agree across grids; 256 rays per grid are also checked against brute force.

Memory is an analytic per-grid-type estimate (element count times `sizeof`, from each grid's
own public accessors), not a tracked allocation total -- it excludes shared geometry/rays,
allocator overhead, and any transient build-time scratch memory. SAH cost constants
(`C_traverse`/`C_intersect`) come from `happyray::MeasureSahCalibration()` (`src/accel/sah_cost.h`):
a handful of timed micro-benchmarks against small synthetic test cases, run once per test
binary invocation. Build/traversal timing uses `std::chrono` wall-clock timing (with a minimal
portable device-to-host read-back forcing real completion on the CUDA backend) and excludes
ray generation and downloads. This file has no CUDA-specific dependency (pure Thrust, like the
rest of the accel/render code), so -- unlike previously -- it builds and runs under **both**
backends; it carries the `performance` label and runs serially.

## Architecture

- `src/core/`: header-only, `HAPPYRAY_HD`-annotated math (`vec.h`, `mat4.h`, `quaternion.h`,
  `random.h`), geometry (`ray.h`, `primitive.h`, `bbox.h`, `triangle_intersect.h`), and scene
  mesh representation (`mesh.h`). No Thrust/CUDA dependency.
  Each `Mesh` stores positions separately from packed `VertexAttributes` (normal and UV);
  both buffers have one entry per vertex and share the same triangle indices. OBJ attribute
  seams retain distinct vertex entries. Flattening extracts the positions and normals used by
  the renderer, whose host/device layout is unchanged.
- `src/accel/`: the uniform grid, the two-level grid, and the irregular grid.
  Each acceleration structure has one header and one implementation file.
  `uniform_grid.h` has shared `HAPPYRAY_HD` build helpers (resolution heuristic, cell-range math,
  triangle/box intersection and occupied-cell extraction)
  used by all three, the `UniformGridView` CSR data view, its `HAPPYRAY_HD`
  3D-DDA traversal (`IntersectSceneGrid`/`IsOccludedGrid`), and the `UniformGrid` builder class
  that produces one; `uniform_grid.cpp` is its sort-based (happyray/HAGrid-style) construction
  algorithm. All three builders keep the final stage inline beside counting, writing and
  sorting fragments: compare adjacent
  sorted cell keys to extract occupied-cell starts, then fill empty-cell CSR offsets with a
  reverse minimum scan (linear work in pairs plus cells, without per-cell binary searches).
  `FragmentCounter` reserves AABB candidates and `FragmentWriter` tests
  triangle/cell overlap using the separating-axis theorem, marking rejects with a sentinel
  key that sorts after all valid cells. A single sentinel lookup determines the valid pair
  count; rejected triangle references and their allocation capacity are released.
  Intersection tests conservatively enlarge cell half-extents by 1% and allow a `1e-5`
  projection tolerance. Unlike the reference's normalized plane test, unnormalized axes
  avoid division by zero for degenerate triangles, and centers remain at true cell centers.
  Candidate ranges and default density parameters are unchanged. Filtering also applies to
  nested grids' top-level construction and initial leaf voxels, before irregular merging.
  `two_level_grid.h`/`.cpp` builds on top of it: a coarse top-level `UniformGrid`
  where *every* cell always recurses into its own nested second-level `UniformGridView` (sparse
  cells just get a degenerate 1x1x1 second-level grid, equivalent to a direct triangle list) --
  this unifies "leaf" and "refined" cells into one traversal with no branching, matching
  happyray's actual `TwoLevelGrid`/`TLGridSortBuilder` design.
  `irregular_grid.h`/`.cpp` build an irregular grid the HAGrid way: each top-level cell gets a
  flat, dense, power-of-2-resolution local grid, then two passes ported from HAGrid's
  `merge.cu`/`expand.cu` refine it bottom-up -- SAH-gated neighbor merging (collapsing empty/
  uniform regions into fewer, bigger cells) and subset-safe directional expansion (growing a
  cell into neighbors whose triangle references are already a subset of its own) -- both scoped
  to stay within one top-level cell's own local grid rather than HAGrid's global scope, which
  sidesteps HAGrid's cross-cell alignment bookkeeping entirely. All three accel structures are expressed purely as
  Thrust functors/algorithm calls (`thrust::transform`, `thrust::for_each`,
  `thrust::exclusive_scan`, `thrust::sort_by_key`, `thrust::lower_bound`, `thrust::scatter`-style
  conditional writes) -- there's one implementation of each, and `HAPPYRAY_THRUST_BACKEND` picks
  whether they get compiled by nvcc (CUDA backend) or plain g++ with
  `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB` (TBB backend).
- `src/render/`: following happyray's engine/integrator separation, `ao_integrator.h` defines
  `AOIntegrator`: AO configuration and host/device `Integrate()` evaluation of a camera ray.
  It owns the AO visibility estimator, material tinting, and Halton hemisphere sampling, but
  no framebuffer or scene allocations. Shared intersection/occlusion queries and
  `AccelStructures`/`AccelType` dispatch live in `src/accel/scene_intersect.h`, independent of
  the integrator. `renderer.h`/`.cpp` is the `Renderer` engine: uploads scene geometry once,
  owns a `UniformGrid`, a `TwoLevelGrid`, and an `IrregularGrid`, and renders each frame via a
  single `thrust::for_each` over a `RenderPixelFunctor` (no `<<<...>>>` kernel launch anywhere).
  `simple_integrator.h` provides deterministic `SimpleIntegrator` cartoon shading with a
  configurable world-space `light_direction` and background color. Both integrators share the
  same host/device `Integrate()` interface. The renderer specializes its shared Thrust pixel
  functor for each type without virtual calls or a per-pixel mode switch.
  `RenderFrame(camera, integrator)` generates camera rays, invokes the supplied integrator,
  accumulates and resolves frames, and downloads pixels. Reset accumulation after changing
  camera or integrator configuration. Both backends use the same implementation.
  `RenderProgressiveFrame(camera, ao, simple)` caches a simple-only preview after each reset,
  then accumulates only AO frames and resolves their gradual blend for the viewer. The
  single-integrator `RenderFrame` path remains unchanged for benchmarks and direct rendering;
  reset before switching between these paths.
- `src/app/`: `obj_loader.h`/`.cpp` (OBJ + `mtllib`/`Kd` parsing), `scene.h` (the shared
  `SceneView`, `FlatScene`, flattening, subdivision, and camera/AO helpers),
  `main.cpp` (the
  interactive SDL2 viewer), and `benchmark.cpp` (the headless `happyray_benchmark` timing harness,
  no SDL dependency).
  The header-only `happyray_scene` CMake target exposes `scene.h` to the loader, renderer,
  acceleration structures, and tests without linking them to an application executable.

Validation strategy: `scene_intersect.h`'s brute-force path (plain, single-threaded, no Thrust at all) is
the project's correctness oracle. `tests/renderer_tests.cpp` renders the same scene through a
dead-simple host reference loop and through `Renderer` (whichever backend the test binary was built
with), checks they agree closely, then switches `Renderer` to the uniform grid, the two-level grid,
and the irregular grid and checks each against the *same* reference again -- since both traces
draw identical deterministic Halton samples against identical geometry, any mismatch points at an
actual bug rather than noise. Each grid is checked against two scenes: `box_on_floor.obj` (too
sparse to exceed `dense_cell_threshold`, exercising only each grid's degenerate path) and
`assets/dense_cube.obj` (a densely tessellated cube, 192 triangles, dense enough that several
top-level cells get real, non-degenerate refinement -- for the irregular grid, this also
exercises real merge, confirmed via instrumentation to collapse 512 initial finest cells down to
296, and real expand, confirmed to grow average cell size from 1.73 to 2.77 finest voxels, with
one cell growing to 27).