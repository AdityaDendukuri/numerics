FetchContent_MakeAvailable(raylib)

# Collect backend sources
set(FLUID_BACKENDS
    backends/seq/fluid.cpp
    backends/seq/heat.cpp)
if(OpenMP_CXX_FOUND)
    list(APPEND FLUID_BACKENDS backends/omp/fluid.cpp backends/omp/heat.cpp)
endif()

add_executable(fluid_sim main.cpp src/fluid.cpp src/heat.cpp ${FLUID_BACKENDS})
target_include_directories(fluid_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/apps)
target_link_libraries(fluid_sim PRIVATE numerics raylib)
target_compile_features(fluid_sim PRIVATE cxx_std_17)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(fluid_sim PRIVATE -g -O0)
else()
    target_compile_options(fluid_sim PRIVATE -O3 -march=native)
endif()
