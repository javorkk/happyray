# Scene I/O for the demo app (OBJ + mtllib loading). Kept as its own library (rather than a
# happyray_core source) since it's application-level content loading, not core engine/math, and
# tests under tests/ also need it to load fixture scenes.
add_library(happyray_obj_loader STATIC obj_loader.cpp)
target_include_directories(happyray_obj_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(happyray_obj_loader PUBLIC happyray_scene)

find_package(PNG 1.6 REQUIRED)
add_library(happyray_screenshot STATIC screenshot.cpp)
target_include_directories(happyray_screenshot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(happyray_screenshot PUBLIC PNG::PNG)

add_executable(happyray main.cpp)
target_link_libraries(happyray PRIVATE happyray_core happyray_render happyray_obj_loader
    happyray_screenshot PkgConfig::SDL2)

# Headless benchmark harness (see benchmark.cpp) -- no SDL dependency, since it never opens a
# window.
add_executable(happyray_benchmark benchmark.cpp)
target_link_libraries(happyray_benchmark PRIVATE happyray_core happyray_render happyray_obj_loader)
