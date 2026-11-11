# HappyRay v2.0

A small ray tracer implemented in [Thrust](https://developer.nvidia.com/thrust).
Currently supports fly-throughs in static or animated .obj scenes and procedural point, sphere, and
Bezier curve scenes in an interactive SDL2 window, and collision detection. 
Implements:

  * [A Parallel Algorithm for Construction of Uniform Grids](https://javor.me/data/gpu_grid_construction.pdf)
  * [Two-Level Grids for Ray Tracing on GPUs](https://javor.me/data/two_level_grids.pdf)
  * [GPU Ray Tracing using Irregular Grids](https://javor.me/data/irregular_grid.pdf)

## Building

Dependencies (CMake 3.25+, SDL2, libpng 1.6+, a GCC host compiler, CUDA and optionally TBB).
CUDA toolkit required for thrust headers even if building for CPU.

If using Rez, enter the build environment before running the commands below:

```sh
rez-env cuda-12.9 cmake-3.31 gcc-11 sdl tbb-2020 -- bash
```

For non-interactive build or test commands, use the same packages with `-- bash -c '<command>'`.

```sh
cd HappyRay
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release # -DHAPPYRAY_THRUST_BACKEND=TBB
cmake --build build --parallel "$(nproc)"
ctest --test-dir build --output-on-failure
```

Switch between GPU and CPU backends using `HAPPYRAY_THRUST_BACKEND=CUDA|TBB` (default `CUDA`).
Use `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to specify GPU architectyre (default `89`). 

### VS Code IntelliSense

Local headers use paths relative to the including file. Same-directory includes use just the
filename; cross-directory includes use paths such as `../core/vec.h`.

For Microsoft's C/C++ extension, configure the project with CMake first, then point IntelliSense
at the generated compilation database in your VS Code workspace settings:

```json
{
    "C_Cpp.default.compileCommands": "${workspaceFolder}/build/compile_commands.json"
}
```

Replace `build` with your actual build directory (for example, `build-tbb` or `build-cuda`).
CMake already enables `CMAKE_EXPORT_COMPILE_COMMANDS`. The database supplies the compiler,
include directories, and defines for each translation unit, including third-party dependencies.
Alternatively, if you use the CMake Tools extension, set
`"C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"` and configure the desired
backend there.

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
./build/happyray assets/box_on_floor.obj [width] [height]
./build/happyray attractor:<points|spheres|curves>[:count] [width] [height]
```

`attractor:...` generates a Lorenz attractor trajectory (default 20,000 samples) as points,
spheres, or Catmull-Rom cubic Bezier curves. Point and curve radii are 2 pixels wide at each
primitive's distance from the camera (`Renderer::SetScreenSpaceRadius`); they are recomputed and
the grid rebuilt only once the camera stops moving.

The window starts at an explicit `[width] [height]` if given; otherwise at the scene's sibling
`<name>.camera.cfg`'s `resX`/`resY` if it has one (see below); otherwise at a 512x512 default.
The window is resizable at runtime (drag an edge/corner) -- the framebuffer and on-screen texture
are reallocated at the new size and accumulation restarts.

Controls: `W/A/S/D` to move, `Q`/`Z` to rise/descend, left-click inside the window to grab the
mouse for looking around (click again to release it), `[`/`]` to halve/double the movement step
size (initially 5% of the scene's bounding-box diagonal), `Space` to reset the camera to its
starting viewpoint, `C` to print the current camera's parameters and window resolution
(happyray's camera.cfg text format, including its `resX`/`resY` fields -- see `scene.h`'s
`LoadCameraConfig`) to the console, `O` to toggle collision detection (see below), and `F1` to
print this control list.
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
Press `P` to pause/resume a keyframe animation. To use the legacy animation format, create an
animation `.cfg` file and pass it to the viewer:

```ini
frame prefix = scenes/ogre/ogre0
frame suffix = .obj
number of frames = 9
frame step = 0.4
camera = ogre.camera.cfg
```

For example, `./build/happyray scenes/ogre/scene.cfg` loads `ogre00.obj` through `ogre08.obj`.
Frame and camera paths in the configuration are relative to the `.cfg` file. The viewer loads the
numbered OBJ frames into memory and advances by `frame step` per rendered frame, interpolating
adjacent geometry; the active acceleration structure is rebuilt as frames advance. Passing an OBJ
directly loads it as a static scene. `HAPPYRAY_SCREENSHOT_PATH=/path/to/out.ppm` and
`HAPPYRAY_SCREENSHOT_FRAME=<n>` dump a PPM screenshot and exit after `<n>` accumulated frames for a
still scene, or `<n>` rendered frames during animation (use a `.png` path to save PNG), and
`HAPPYRAY_ACCEL_TYPE=uniform_grid`, `HAPPYRAY_ACCEL_TYPE=two_level_grid`, or
`HAPPYRAY_ACCEL_TYPE=irregular_grid` selects that structure from startup instead of the default, for
headless verification. `HAPPYRAY_CELL_OVERLAP=aabb|exact|hybrid` overrides how grids assign
primitives to cells (see Architecture). A scene can also have a sibling `<name>.camera.cfg` giving it a fixed
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

### Collision detection

Every OBJ `o`/`g` statement starts an object: each distinct (`o` name, `g` name) pair with faces
gets its own id (a new `o` resets the group; repeating a name continues the earlier object). For
scenes with at least two objects, the renderer runs collision detection whenever it builds
an acceleration structure (or switches to an already-built one), including on every animation
frame. This follows string2shape's `CollisionDetector`. Leaf cells that hold primitives from only
one object are dropped. The remaining cells' references are copied out and sorted by (cell,
object), leaving the grid untouched, so pairs from the same object are never visited. In each
remaining cell, every pair of primitives from different objects
collides if their bounding boxes overlap, which works for every primitive type (for box
primitives the test is exact). Each primitive
records the smallest id of another object it collides with, and `CollisionStats` also counts the
pair tests executed and how many found an overlap. Colliding primitives are drawn with a
red "collided" material. For such scenes the window title shows the latest detection time as
`coll: <ms>`; the `build: <ms>` figure excludes it. While an animation is paused (`P`), its
geometry doesn't change, so detection is skipped (density-search rebuilds and grid switches
included) and the last results stay on screen. `O` turns collision detection off (removing the
red highlighting) and back on (detecting immediately on the active grid, even while paused).

## Running the benchmark

```sh
./build/src/app/happyray_benchmark [scene.obj|attractor:<points|spheres|curves>[:count] ...]
                               [--width W] [--height H] [--frames N] [--warmup N] [--samples N]
                               [--assets-dir DIR] [--overlap aabb|exact|hybrid]
                               [--radius PIXELS] [--no-brute-force]
```

Headless (no SDL/window dependency): times `Renderer::RenderFrame` for every acceleration
structure (brute force, uniform grid, two-level grid) against each given scene (or, with no
scene arguments, `assets/box_on_floor.obj` and `assets/dense_cube.obj`), after a few untimed
warmup frames, and prints a table of mean/min/max frame time, estimated AO-ray throughput
(Mrays/s), and grid build time. Like the viewer, it only ever exercises whichever Thrust backend it was built with —
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

### Collision performance test

`collision_performance_fairy` benchmarks collision detection on the moving geometry of frames
105 to 106 of the animated Fairy Forest, interpolated from `scenes/fairy/f100.obj` and `f110.obj`
(`scenes/` is untracked; extract `scenes.7z` first -- the test is only registered if both files
exist). Each triangle is replaced by the box bounding it at both frames (`MakeSweptTriangleBoxes`
in `scene.h`), and each grid is built over these boxes as `PrimitiveType::kBox` primitives,
with densities tuned per backend for this benchmark (found with `--search-densities`, below;
the renderer's defaults are unchanged). The
renderer's `CollisionDetector::ComputeCollisions` then runs on each grid. Per grid, the table
reports total, build, and collision time (mean of 5 runs after one warmup), candidate pairs (box
pairs from different objects the detector actually tested, counting a pair once per shared cell
and excluding tests skipped because they could not change a recorded result), and colliding
pairs (tests that found overlapping boxes). Both counts measure the work done, so they vary
slightly between runs and backends. Per-box collision results must match a sweep-and-prune
oracle (a parallel sort by minimum x, then every box's sweep split into fixed-size chunks
tested in parallel, in Thrust on the same CUDA or TBB backend as the grids); its runtime, with the same warmup, averaging, and persistent temporary
storage as the detector, is the "Baseline ms" column. Timings are informational.

```sh
ctest --test-dir build -R collision_performance -V
```

Run the test binary with `--search-densities` (not part of CTest) to additionally search each
grid's densities for the lowest total time: a 2x-spaced coarse sweep followed by a hill climb
over half-octave (sqrt(2)) steps, then the best five and the benchmark's current densities
re-measured with 15 runs.

```sh
./build/tests/happyray_collision_performance_tests scenes/fairy --search-densities
```

## Architecture

- `src/core/`: header-only, `HAPPYRAY_HD`-annotated math (`vec.h`, `mat4.h`, `quaternion.h`,
  `random.h`), geometry (`ray.h`, `primitive.h`, `point.h`, `sphere.h`, `bezier_curve.h`,
  `bbox.h`, `triangle.h`), and scene mesh representation (`mesh.h`). Triangle, point, sphere,
  cubic Bezier tube, and solid box (`BBox` as `PrimitiveType::kBox`, held in the `boxes` arrays
  of `SceneView`/`FlatScene`) geometry includes ray and box intersection predicates; triangle
  pairs also have an intersection test.
  Each `Mesh` stores positions separately from packed `VertexAttributes` (normal and UV);
  both buffers have one entry per vertex and share the same triangle indices. OBJ attribute
  seams retain distinct vertex entries. Flattening extracts the positions and normals used by
  the renderer, whose host/device layout is unchanged.
  `primitive_traits.h` defines both hit records (`ClosestHit`, `HitRecord`) and the
  per-primitive-type interface (`PrimitiveTraits<P>`: fetch, ray intersection, shading normal,
  and exact cell overlap) for triangles, points, spheres, and Bezier curves. The traits use
  the application's `SceneView` from `scene.h`.
- `src/accel/`: the uniform grid, the two-level grid, and the irregular grid.
  Each acceleration structure has one header and one implementation file.
  Grid views are primitive-agnostic; builders and leaf intersection are templated on the
  primitive type and dispatched once per build or kernel launch from `SceneView::primitive_type`.
  Shared `IntersectSceneGrid<P>` and `IsOccludedGrid<P>` templates in `scene_intersect.h`
  accept any grid view. They default to `DynamicPrimitive`, which dispatches at runtime.
  `CellOverlapMode` selects how builders assign primitives to cells: `kAabb` (every cell of the
  primitive's AABB), `kExact` (exact, conservative primitive/cell tests for counting and writing;
  the CUDA default), or `kHybrid` (AABB counting, exact writing, trimmed padding; the TBB default).
  `uniform_grid.h` provides shared resolution, cell-range, and primitive/cell overlap helpers.
  `UniformGridView` stores one begin/end pair per cell, and its `HAPPYRAY_HD` 3D-DDA traversal
  (`IntersectSceneGrid`/`IsOccludedGrid`) tests the references in that interval.
  The backend-private `grid_common.h` shares the complete coarse construction operation:
  bounds, count, scan, emit, sort, and range extraction, specialized per overlap mode.
  Empty cells keep equal zero indices, with no CSR gap-fill scan.
  `uniform_grid.cpp` owns the resulting buffers. `irregular_grid.cpp` uses UniformGrid for
  its top-level layout, then builds a separate local voxel CSR, which its merge/expand passes
  refine. `two_level_grid.cpp` consumes the shared coarse build's sorted keys and ranges,
  then builds the second level directly; it uses the same exact SAT test and explicit begin/end ranges at both levels,
  with cubic per-top-cell resolutions and 1x1x1 grids for sparse cells, following the legacy
  `TLGridSortBuilder`/`TLGridBuildKernels` model. The SAT test conservatively enlarges cell
  half-extents by 1% and uses a `1e-5` projection tolerance; unnormalized axes handle degenerate
  triangles without division by zero. These builders use Thrust algorithms and compile for the
  CUDA and TBB backends selected by `HAPPYRAY_THRUST_BACKEND`.
  Traversal keeps one explicit loop per algorithm (nested loops for two-level grids).
  Uniform and two-level grids deliberately keep independent DDA initialization and stepping
  so they can be optimized separately. Each selects the next axis using the legacy
  minimum-dimension expression and uses a `switch` with constant indices into three-element
  cell/boundary arrays so the compiler can keep state in registers rather than memory.
  Zero-direction axes never step, and ties
  retain the legacy z/y/x priority.   All three overload `detail::TraverseGrid` and pass a `CellRange` to shared closest-hit
  or occlusion visitors; SAH calibration uses the same loops with counting visitors.
  Counting and writing fragments share geometry setup through compile-time specialization.
  `irregular_grid.h`/`.cpp` build an irregular grid the HAGrid way: each top-level cell gets a
  flat, dense, power-of-2-resolution local grid, then two passes ported from HAGrid's
  `merge.cu`/`expand.cu` refine it bottom-up -- SAH-gated neighbor merging (collapsing empty/
  uniform regions into fewer, bigger cells) and subset-safe directional expansion (growing a
  cell into neighbors whose triangle references are already a subset of its own) -- both scoped
  to stay within one top-level cell's own local grid rather than HAGrid's global scope, which
  sidesteps HAGrid's cross-cell alignment bookkeeping entirely. All three accel structures are expressed purely as
  All three builders use Thrust functors and algorithms; the uniform and two-level grids extract
  cell ranges directly from sorted keys, while the irregular builder retains CSR for its local
  voxels. `HAPPYRAY_THRUST_BACKEND` selects CUDA (nvcc) or TBB (plain g++) execution.
- `src/render/`: following happyray's engine/integrator separation, `ao_integrator.h` defines
  `AOIntegrator`: AO configuration and host/device `Integrate()` evaluation of a camera ray.
  It owns the AO visibility estimator, material tinting, and Halton hemisphere sampling, but
  no framebuffer or scene allocations. Shared intersection/occlusion queries and
  `AccelStructures`/`AccelType` dispatch live in `src/accel/scene_intersect.h`, independent of
  the integrator. `renderer.h`/`.cpp` is the `Renderer` engine: uploads scene geometry once,
  owns a `UniformGrid`, a `TwoLevelGrid`, and an `IrregularGrid`, and renders each frame via a
  single `thrust::for_each` over a `RenderPixelFunctor` (no `<<<...>>>` kernel launch anywhere).
  `SetAccelType(type)` selects a grid without building it; call
  `BuildAccelerationStructure(settings)` explicitly before rendering a missing grid.
  `HasAccelerationStructure()` reports whether the selected structure is ready (always true
  for brute force, whose build is a no-op). Rendering an unbuilt grid throws.
  `GridBuildSettings` supplies densities (target cells per primitive, per grid level; higher
  means more, smaller cells), the dense-cell threshold, and irregular-grid
  merge/expand parameters. Zero densities use renderer defaults: `{3, 0}` for uniform,
  `{0.12, 2.4}` for nested grids. Scene updates and density tuning retain each grid's settings;
  an explicit build with omitted settings uses defaults again. For example:

  ```cpp
  renderer.SetAccelType(happyray::AccelType::kTwoLevelGrid);
  renderer.BuildAccelerationStructure({{0.12f, 1.2f}, 16});
  ```

  `collision_detector.h`/`.cpp` reuse the leaf cells of any of the three grids for bounding-box
  object/object collision detection (see "Collision detection" above), parallelized per cell
  reference with Thrust; the renderer runs it after every grid build. A `CollisionDetector` keeps
  its temporary buffers and Thrust's scratch memory (through the caching allocator in
  `src/core/thrust_scratch.h`) between calls, so repeated detections don't allocate; it is not
  thread-safe.
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
- `src/app/`: `obj_loader.h`/`.cpp` (OBJ + `mtllib`/`Kd` parsing, `o`/`g` object ids), `scene.h` (the shared
  `SceneView`, `FlatScene`, flattening, subdivision, and camera/AO helpers), `attractor_scene.h`
  (procedural point/sphere/curve scenes),
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
one cell growing to 27).Triangle, point, sphere, and curve scenes are additionally rendered through every grid in every
cell-overlap mode and compared against the renderer's own brute-force output.
