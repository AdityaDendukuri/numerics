FetchContent_MakeAvailable(raylib)

add_executable(ns_demo main.cpp src/sim.cpp)
target_include_directories(ns_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/apps)
target_link_libraries(ns_demo PRIVATE numerics raylib)
target_compile_features(ns_demo PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ns_demo PRIVATE -g -O0)
else()
    target_compile_options(ns_demo PRIVATE -O3 -march=native)
endif()
