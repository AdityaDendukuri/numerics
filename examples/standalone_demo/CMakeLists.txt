cmake_minimum_required(VERSION 3.20)
project(standalone_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find installed numerics package
find_package(numerics REQUIRED)
include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(standalone_demo main.cpp)
target_link_libraries(standalone_demo PRIVATE numerics::numerics nlohmann_json::nlohmann_json)
