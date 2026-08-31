cmake_minimum_required(VERSION 3.20)
project(numerics_core_consumer LANGUAGES CXX)

find_package(numerics REQUIRED COMPONENTS core CONFIG)

add_executable(core_consumer main.cpp)
target_link_libraries(core_consumer PRIVATE numerics::core)
target_compile_features(core_consumer PRIVATE cxx_std_20)
