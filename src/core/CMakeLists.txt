# Header-only: vec/mat4/ray/bbox/primitive/quaternion/random/mesh are all
# inline HAPPYRAY_HD headers with no .cpp of their own (obj_loader lives in src/app -- it's scene
# I/O for the demo app, not core engine/math).
add_library(happyray_core INTERFACE)
target_include_directories(happyray_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
