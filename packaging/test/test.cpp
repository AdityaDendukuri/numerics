#include <numerics.hpp>

#include <iostream>

int main() {
    num::vec<num::real> x{3.0, 4.0};
    std::cout << num::norm(x) << "\n";
}
