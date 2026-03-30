FetchContent_MakeAvailable(raylib)

add_executable(ising_nucleation main.cpp)
target_include_directories(ising_nucleation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/apps
    ${CMAKE_SOURCE_DIR}/apps/ising/include)
target_link_libraries(ising_nucleation PRIVATE numerics raylib)
target_compile_features(ising_nucleation PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ising_nucleation PRIVATE -g -O0)
else()
    target_compile_options(ising_nucleation PRIVATE -O3 -march=native)
endif()
