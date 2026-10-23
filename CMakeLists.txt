cmake_minimum_required(VERSION 3.25)

project(HappyRay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-O3>)
add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-march=native>)

# Selects which Thrust backend parallelizes the renderer/uniform grid (src/render/renderer.cpp,
# src/accel/uniform_grid/uniform_grid.cpp): CUDA runs them on the GPU (needs nvcc); TBB runs the
# exact same Thrust functors/algorithms in parallel on the CPU instead (plain g++, no CUDA
# compiler needed at all). Each build only supports the backend it was configured with -- one
# binary per backend, not a runtime toggle.
set(HAPPYRAY_THRUST_BACKEND "CUDA" CACHE STRING "Thrust backend: CUDA or TBB")
set_property(CACHE HAPPYRAY_THRUST_BACKEND PROPERTY STRINGS CUDA TBB)

if(HAPPYRAY_THRUST_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # NVIDIA L4 (Ada Lovelace, SM 8.9) is the primary target GPU in the dev environment;
    # override with -DCMAKE_CUDA_ARCHITECTURES=... for other hardware.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    # Needed for uniform_grid_view.h's grid-traversal lambdas to compile as device code.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
elseif(NOT HAPPYRAY_THRUST_BACKEND STREQUAL "TBB")
    message(FATAL_ERROR "HAPPYRAY_THRUST_BACKEND must be CUDA or TBB (got '${HAPPYRAY_THRUST_BACKEND}')")
endif()

# Thrust ships as headers inside the CUDA toolkit's include tree, so this is needed even for a
# TBB-backend build (which otherwise never invokes nvcc or links the CUDA runtime).
find_package(CUDAToolkit REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)

# Carries whichever compile definitions/include dirs/link libraries the active Thrust backend
# needs. src/accel and src/render link this into their backend-switched source files
# (uniform_grid.cpp, renderer.cpp) instead of duplicating this logic in each CMakeLists.txt.
add_library(happyray_thrust_backend INTERFACE)
target_link_libraries(happyray_thrust_backend INTERFACE CUDA::toolkit)
if(HAPPYRAY_THRUST_BACKEND STREQUAL "CUDA")
    target_link_libraries(happyray_thrust_backend INTERFACE CUDA::cudart)
else()
    # Prefer system TBB (e.g. Ubuntu's libtbb-dev); older Rez TBB packages may not
    # provide a CMake config package, so retain their REZ_TBB_ROOT fallback.
    find_package(TBB CONFIG QUIET)
    if(TARGET TBB::tbb)
        target_link_libraries(happyray_thrust_backend INTERFACE TBB::tbb)
    elseif(DEFINED ENV{REZ_TBB_ROOT})
        target_include_directories(happyray_thrust_backend INTERFACE "$ENV{REZ_TBB_ROOT}/include")
        target_link_directories(happyray_thrust_backend INTERFACE "$ENV{REZ_TBB_ROOT}/lib64")
        target_link_libraries(happyray_thrust_backend INTERFACE tbb)
    else()
        message(FATAL_ERROR
            "TBB not found: install a system TBB development package or set REZ_TBB_ROOT.")
    endif()
    target_compile_definitions(happyray_thrust_backend INTERFACE
        TBB_SUPPRESS_DEPRECATED_MESSAGES
        HAPPYRAY_THRUST_BACKEND_IS_TBB=1)
endif()

enable_testing()

add_subdirectory(src/core)

add_library(happyray_scene INTERFACE)
target_include_directories(happyray_scene INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/app)
target_link_libraries(happyray_scene INTERFACE happyray_core)

add_subdirectory(src/accel)
add_subdirectory(src/render)
add_subdirectory(src/app)
add_subdirectory(tests)
