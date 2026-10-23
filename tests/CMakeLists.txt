add_executable(happyray_sdl_test sdl_test.cpp)
target_link_libraries(happyray_sdl_test PRIVATE PkgConfig::SDL2)
add_test(NAME sdl_test COMMAND happyray_sdl_test)

add_executable(happyray_screenshot_tests screenshot_tests.cpp)
target_link_libraries(happyray_screenshot_tests PRIVATE happyray_screenshot)
add_test(NAME screenshot_tests COMMAND happyray_screenshot_tests)

add_executable(happyray_core_tests core_tests.cpp)
target_link_libraries(happyray_core_tests PRIVATE happyray_core happyray_obj_loader)

add_test(NAME core_tests COMMAND happyray_core_tests "${CMAKE_SOURCE_DIR}/assets")

add_executable(happyray_render_tests render_tests.cpp)
target_link_libraries(happyray_render_tests PRIVATE happyray_core happyray_render happyray_obj_loader)

add_test(NAME render_tests COMMAND happyray_render_tests "${CMAKE_SOURCE_DIR}/assets")

add_executable(happyray_renderer_tests renderer_tests.cpp)
target_link_libraries(happyray_renderer_tests PRIVATE happyray_core happyray_render happyray_obj_loader)
# renderer_tests.cpp's irregular-grid infinite-loop regression check uses a watchdog thread
# (std::thread) to bound how long it waits for a potentially-hung render before failing loudly.
find_package(Threads REQUIRED)
target_link_libraries(happyray_renderer_tests PRIVATE Threads::Threads)

add_test(NAME renderer_tests COMMAND happyray_renderer_tests "${CMAKE_SOURCE_DIR}/assets")

add_executable(happyray_grid_build_tests grid_build_tests.cpp)
target_link_libraries(happyray_grid_build_tests PRIVATE happyray_accel happyray_thrust_backend)

# Pure Thrust (no raw CUDA runtime API), same as grid_build_tests.cpp above: built as actual CUDA
# source under the CUDA backend (nvcc), or as plain C++ under the TBB backend (g++), so both
# binaries -- and the SAH-cost/calibration measurements they report -- exist regardless of which
# backend this project was configured with.
add_executable(happyray_accel_performance_tests accel_performance_tests.cpp)
target_link_libraries(happyray_accel_performance_tests PRIVATE
    happyray_accel happyray_render happyray_obj_loader happyray_thrust_backend)
foreach(scene fairy conference ogre)
    add_test(NAME accel_performance_${scene} COMMAND happyray_accel_performance_tests
        "${CMAKE_SOURCE_DIR}/assets/${scene}.obj")
    set_tests_properties(accel_performance_${scene}
        PROPERTIES RUN_SERIAL TRUE LABELS "performance" TIMEOUT 300)
endforeach()

if(HAPPYRAY_THRUST_BACKEND STREQUAL "CUDA")
    set_source_files_properties(grid_build_tests.cpp accel_performance_tests.cpp
        PROPERTIES LANGUAGE CUDA)
endif()
add_test(NAME grid_build_tests COMMAND happyray_grid_build_tests)