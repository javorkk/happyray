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

# CUDA runs the Thrust code on the GPU; TBB runs the same code on the CPU without nvcc.
set(HAPPYRAY_THRUST_BACKEND "CUDA" CACHE STRING "Thrust backend: CUDA or TBB")
set_property(CACHE HAPPYRAY_THRUST_BACKEND PROPERTY STRINGS CUDA TBB)

if(HAPPYRAY_THRUST_BACKEND STREQUAL "CUDA")
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)  # NVIDIA A4000
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Device lambdas in the traversal code.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Wno-deprecated-gpu-targets>)
elseif(NOT HAPPYRAY_THRUST_BACKEND STREQUAL "TBB")
    message(FATAL_ERROR "HAPPYRAY_THRUST_BACKEND must be CUDA or TBB (got '${HAPPYRAY_THRUST_BACKEND}')")
endif()

# Thrust headers come from the CUDA toolkit, even for TBB builds.
find_package(CUDAToolkit REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)

# Compile definitions, include dirs, and libraries for the active Thrust backend.
add_library(happyray_thrust_backend INTERFACE)
target_link_libraries(happyray_thrust_backend INTERFACE CUDA::toolkit)
if(HAPPYRAY_THRUST_BACKEND STREQUAL "CUDA")
    target_link_libraries(happyray_thrust_backend INTERFACE CUDA::cudart)
else()
    # Older Rez TBB packages lack a CMake config; fall back to REZ_TBB_ROOT.
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
