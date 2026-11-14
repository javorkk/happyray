# Scene loading, a separate library because the tests also load fixtures with it.
add_library(happyray_obj_loader STATIC obj_loader.cpp)
target_include_directories(happyray_obj_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(happyray_obj_loader PUBLIC happyray_scene)

# OpenUSD is optional; without it the loader is a stub that reports missing support.
option(HAPPYRAY_ENABLE_USD "Load .usd/.usda/.usdc/.usdz scenes and animations through OpenUSD" OFF)
add_library(happyray_usd_loader STATIC usd_loader.cpp)
target_include_directories(happyray_usd_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(happyray_usd_loader PUBLIC happyray_scene)
if(HAPPYRAY_ENABLE_USD)
    # Rez's usdBase package ships pxrConfig.cmake in its root rather than on CMAKE_PREFIX_PATH.
    find_package(pxr CONFIG REQUIRED PATHS "$ENV{REZ_USDBASE_ROOT}")
    find_package(Threads REQUIRED)
    target_compile_definitions(happyray_usd_loader PUBLIC HAPPYRAY_WITH_USD=1)
    target_link_libraries(happyray_usd_loader PUBLIC usdGeom usd Threads::Threads)
    if(DEFINED usdBase_LIBRARIES)
        # The facility's usdBase headers pull in boost-python and Python, which the imported
        # usd targets above do not carry.
        target_link_directories(happyray_usd_loader PUBLIC ${usdBase_LIBRARY_DIRS})
        target_link_libraries(happyray_usd_loader PUBLIC ${usdBase_LIBRARIES})
    endif()
endif()

find_package(PNG 1.6 REQUIRED)
add_library(happyray_screenshot STATIC screenshot.cpp)
target_include_directories(happyray_screenshot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(happyray_screenshot PUBLIC PNG::PNG)

add_executable(happyray main.cpp)
set_target_properties(happyray PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(happyray PRIVATE happyray_core happyray_render happyray_obj_loader
    happyray_usd_loader happyray_screenshot PkgConfig::SDL2)

add_executable(happyray_benchmark benchmark.cpp)
target_link_libraries(happyray_benchmark PRIVATE happyray_core happyray_render happyray_obj_loader)
