FetchContent_MakeAvailable(raylib)

add_executable(nbody main.cpp src/sim.cpp)
target_include_directories(nbody PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/apps)
target_link_libraries(nbody PRIVATE numerics raylib)
target_compile_features(nbody PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(nbody PRIVATE -g -O0)
else()
    target_compile_options(nbody PRIVATE -O3 -march=native)
endif()
