add_executable(lorenz lorenz.cpp)
target_link_libraries(lorenz PRIVATE numerics)

add_executable(fft_demo fft_demo.cpp)
target_link_libraries(fft_demo PRIVATE numerics)

add_executable(ising_demo ising_demo.cpp)
target_link_libraries(ising_demo PRIVATE numerics)
