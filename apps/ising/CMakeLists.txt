FetchContent_MakeAvailable(raylib)

add_executable(ising main.cpp)
target_include_directories(ising PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/apps)
target_link_libraries(ising PRIVATE numerics raylib)
target_compile_features(ising PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ising PRIVATE -g -O0)
else()
    target_compile_options(ising PRIVATE -O3 -march=native)
endif()
