cmake_minimum_required(VERSION 3.20)
project(standalone_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find installed numerics package
find_package(numerics REQUIRED)

add_executable(standalone_demo main.cpp)
target_link_libraries(standalone_demo PRIVATE numerics::numerics)
