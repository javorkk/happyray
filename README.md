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
rez-env cuda-12.9 cmake-3.31 gcc-11 sdl tbb-2020 usdBase-0.24.3.1.4 -- bash
```

For non-interactive build or test commands, use the same packages with `-- bash -c '<command>'`.

```sh
cd HappyRay
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release # -DHAPPYRAY_THRUST_BACKEND=TBB -DHAPPYRAY_ENABLE_USD=ON
cmake --build build --parallel "$(nproc)"
ctest --test-dir build --output-on-failure
```

Switch between GPU and CPU backends using `HAPPYRAY_THRUST_BACKEND=CUDA|TBB` (default `CUDA`).
Use `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to specify GPU architecture (default `86`).


### Docker

The [Dockerfile](Dockerfile) builds a development image with the
[VFX Reference Platform CY2026](https://vfxplatform.com/) toolchain: CUDA 12.9 on Ubuntu 24.04,
gcc 14.2 and oneTBB 2022.3, plus the OpenUSD 26.08 core libraries (no Python or imaging) in
`/opt/openusd`, found through `CMAKE_PREFIX_PATH`.

```sh
docker build -t happyray-dev .
docker run --rm -it --gpus all -v "$PWD":/app -w /app happyray-dev bash
# inside the container:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHAPPYRAY_ENABLE_USD=ON
cmake --build build --parallel "$(nproc)"
```

Build args: `ONETBB_VERSION`, `OPENUSD_VERSION`, and `BUILD_JOBS` (default 4, since OpenUSD needs
2-3 GB per compiler process). Dependency builds are retried up to 3 times to work around
occasional gcc crashes under WSL2.

### VS Code IntelliSense

CMake exports `compile_commands.json`; point the C/C++ extension at it in your workspace settings
(replace `build` with your build directory):

```json
{
    "C_Cpp.default.compileCommands": "${workspaceFolder}/build/compile_commands.json"
}
```

## X forwarding using Docker on WSL2

In WSL2:
```sh
touch ~/.Xauthority
xauth add ${HOST}:0 . $(xxd -l 16 -p /dev/urandom)
xauth list   # e.g. IGOR-II/unix:0  MIT-MAGIC-COOKIE-1  9e6238034158c72ba4b08487c0ff0cd0
```
Then, in the container, add the listed cookie:
```sh
xauth add ${DISPLAY} MIT-MAGIC-COOKIE-1  9e6238034158c72ba4b08487c0ff0cd0
```

## Running the viewer

```sh
./build/happyray assets/box_on_floor.obj [width] [height]
./build/happyray attractor:<points|spheres|curves>[:count] [width] [height]
./build/happyray scenes/ogre/scene.cfg [width] [height]
```

`attractor:...` generates a Lorenz attractor trajectory (default 20,000 samples) as points,
spheres, or cubic Bezier curves. Point and curve radii are 2 pixels at each primitive's distance
from the camera and are recomputed once the camera stops.

A scene `<name>.obj` may have a sibling `<name>.camera.cfg` with the starting viewpoint and
resolution (`resX`/`resY`). The window size is taken from the command line, else that file, else
512x512, and can be resized at runtime.

Controls (`F1` prints them):

| Key | Action |
| --- | --- |
| `W`/`A`/`S`/`D`, `Q`/`Z` | Move; rise/descend |
| Left click | Grab/release the mouse for looking around |
| Mouse wheel | Look sensitivity (start value: `HAPPYRAY_LOOK_SENSITIVITY`, default `0.1`) |
| `[` / `]` | Halve/double the step size (initially 5% of the scene diagonal) |
| `Space` | Reset the camera |
| `C` | Print the camera in `camera.cfg` format |
| `1` / `2` / `3` | Uniform / two-level (default) / irregular grid |
| `O` | Toggle collision detection |
| `P` | Pause/resume animation |
| `T` | Save `happyray_<date>_<time>.png` in the working directory |
| `Esc` | Quit |

An animation `.cfg` plays numbered OBJ keyframes (paths relative to the `.cfg`), interpolating
between them by `frame step` per rendered frame and rebuilding the grid as it goes:

```ini
frame prefix = scenes/ogre/ogre0
frame suffix = .obj
number of frames = 9
frame step = 0.4
camera = ogre.camera.cfg
```

Environment variables for headless runs: `HAPPYRAY_SCREENSHOT_PATH=<file.ppm|file.png>` with
`HAPPYRAY_SCREENSHOT_FRAME=<n>` saves a screenshot after `n` frames and exits;
`HAPPYRAY_ACCEL_TYPE=uniform_grid|two_level_grid|irregular_grid` selects the starting grid;
`HAPPYRAY_CELL_OVERLAP=aabb|exact|hybrid` overrides how grids assign primitives to cells.

The first frame after a camera move uses simple cartoon shading; after `n` AO frames the
display is `(1 - w) * simple + w * average_AO` with `w = n / (n + 32)`.

After 32 still frames, the viewer searches the active grid's densities in `[0.25x, 4x]` the
defaults, keeping the candidate with the lowest SAH cost (`src/accel/sah_cost.h`). SAH costs need
no rays: they come from the grid's cell layout and traversal/intersection costs measured once at
startup. The search is disabled for scenes with more than 1,000,000 primitives.

### USD scenes and animations

With `-DHAPPYRAY_ENABLE_USD=ON` (under Rez, add `usdBase-0.24.3.1.4` to the `rez-env` packages)
the viewer also loads `.usd`/`.usda`/`.usdc`/`.usdz` files. Visible meshes are triangulated with
their world transforms baked in (Z-up stages become Y-up) and colored by `displayColor`; points
and basis curves load as native points and Bezier curves. A stage must contain a single primitive
type. Time samples become keyframes that play back like an OBJ animation; the primitive set is
fixed by the first frame and point counts must stay constant.

An animation `.cfg` can subsample or add a camera (`camera` defaults to
`<usd name>.camera.cfg`):

```ini
usd = my_scene.usdc
start time = 951     # optional, inclusive, in stage time codes
end time = 1188      # optional, inclusive
frame stride = 2     # optional, keep every 2nd time sample
frame step = 0.5
camera = my_scene.camera.cfg
```

### Collision detection

Each distinct OBJ (`o`, `g`) name pair with faces is one object. For scenes with at least two
objects, every grid build (or switch to a built grid) runs collision detection, following
string2shape's `CollisionDetector`: leaf cells holding a single object are dropped, and primitives
from different objects sharing a cell collide if their bounding boxes overlap. Colliding
primitives are drawn red, and the window title shows the detection time as `coll: <ms>`
(excluded from `build: <ms>`). Detection is skipped while an animation is paused.

## Running the benchmark

```sh
./build/src/app/happyray_benchmark [scene.obj|attractor:<points|spheres|curves>[:count] ...]
                               [--width W] [--height H] [--frames N] [--warmup N] [--samples N]
                               [--assets-dir DIR] [--overlap aabb|exact|hybrid]
                               [--radius PIXELS] [--no-brute-force]
```

Times `Renderer::RenderFrame` for brute force and the three grids on each scene (default
`assets/box_on_floor.obj` and `assets/dense_cube.obj`) and prints frame times, AO-ray throughput,
and grid build time. It measures the backend it was built with.

### Acceleration performance tests

```sh
ctest --test-dir build -L performance -V   # -V shows the tables of passing tests
```

For Fairy Forest, Conference, and Ogre, each grid is built once after a warm-up build and traces
1,048,576 random closest-hit rays. Tables report build time, ray throughput, analytic memory
footprint (persistent buffers only), and SAH cost. Orderings between grids are informational;
only wrong hits (checked across grids and against brute force) or errors fail. Scenes are
subdivided on load until no edge exceeds 0.1x the scene diagonal.

### Collision performance test

`collision_performance_fairy` detects collisions between the swept triangle bounding boxes of
Fairy Forest frames 105-106 (interpolated from `scenes/fairy/f100.obj` and `f110.obj`; extract
`scenes.7z` first, otherwise the test is not registered). Per grid it reports build, collision,
and total time (mean of 5 runs) and pair-test counts. Results must match a sweep-and-prune
oracle, whose time is the "Baseline ms" column.

```sh
ctest --test-dir build -R collision_performance -V
./build/tests/happyray_collision_performance_tests scenes/fairy --search-densities
```

`--search-densities` additionally searches each grid's densities for the lowest total time.

## Architecture

- `src/core/`: header-only host/device math, primitives (triangle, point, sphere, cubic Bezier
  curve, solid box), and `PrimitiveTraits<P>` (fetch, ray intersection, shading normal, cell
  overlap). Triangles are traced through Woop's unit-triangle records (`TriangleWoop`).
- `src/accel/`: the uniform, two-level, and irregular grids, built with sort-based Thrust
  algorithms for either backend. Builders and leaf tests are templated on the primitive type.
  `CellOverlapMode` selects cell assignment: `kAabb`, `kExact` (CUDA default), or `kHybrid`
  (AABB counting, exact writing; TBB default). The irregular grid refines per-top-level-cell
  power-of-2 local grids with HAGrid's merge and expand passes, which may cross top-level cells.
  `intersect_common.h` holds the shared queries; brute force is the correctness oracle.
- `src/render/`: `Renderer` uploads the scene, owns the grids, and renders with Thrust functors
  specialized per integrator (`AOIntegrator`, `SimpleIntegrator`), primitive type, and grid.
  `CollisionDetector` reuses grid leaf cells for collision detection. Grids must be built
  explicitly before rendering:

  ```cpp
  renderer.SetAccelType(happyray::AccelType::kTwoLevelGrid);
  renderer.BuildAccelerationStructure({{0.12f, 1.2f}, 16}); // densities (cells per primitive), dense-cell threshold
  ```

  Zero densities select the defaults: `{3, 0}` for the uniform grid, `{0.12, 2.4}` for nested grids.
- `src/app/`: OBJ/USD loaders, `scene.h` (scene views and flattening), the procedural attractor
  scenes, the SDL2 viewer (`main.cpp`), and the benchmark.
