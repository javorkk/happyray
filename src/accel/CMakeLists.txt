add_library(happyray_accel STATIC
    uniform_grid.cpp
    two_level_grid.cpp
    irregular_grid.cpp
    sah_cost.cpp
)

if(HAPPYRAY_THRUST_BACKEND STREQUAL "CUDA")
    set_source_files_properties(uniform_grid.cpp two_level_grid.cpp irregular_grid.cpp sah_cost.cpp
                                PROPERTIES LANGUAGE CUDA)
    set_target_properties(happyray_accel PROPERTIES CUDA_STANDARD 17)
endif()

target_include_directories(happyray_accel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(happyray_accel PUBLIC happyray_scene)
target_link_libraries(happyray_accel PRIVATE happyray_thrust_backend)
