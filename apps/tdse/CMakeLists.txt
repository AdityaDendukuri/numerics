FetchContent_MakeAvailable(raylib)

add_executable(tdse main.cpp src/sim.cpp)
target_include_directories(tdse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/apps)
target_link_libraries(tdse PRIVATE numerics raylib)
target_compile_features(tdse PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(tdse PRIVATE -g -O0)
else()
    target_compile_options(tdse PRIVATE -O3 -march=native)
endif()
