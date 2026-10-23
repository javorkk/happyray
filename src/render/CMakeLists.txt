add_library(happyray_render STATIC
    renderer.cpp
)

if(HAPPYRAY_THRUST_BACKEND STREQUAL "CUDA")
    set_source_files_properties(renderer.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(happyray_render PROPERTIES CUDA_STANDARD 17)
endif()

target_include_directories(happyray_render PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(happyray_render PUBLIC happyray_core happyray_accel)
target_link_libraries(happyray_render PRIVATE happyray_thrust_backend)
