# Consumer check for an installed numerics package: find it, link it, run it.
cmake_minimum_required(VERSION 3.20)
project(numerics_package_test CXX)
find_package(numerics 1.1 REQUIRED)
add_executable(test_numerics test.cpp)
target_link_libraries(test_numerics PRIVATE numerics::numerics)
