/// @file 14_concepts_and_property_invariants.cpp
/// @brief The SPD law, how to attach it, and what happens without it.
///
/// `cholesky(A)` does not compile unless A carries the SPD law. `assume_spd(A)` attaches the
/// law and, at the active diagnostic level, tests it on a few random vectors. `make_spd(A)`
/// checks it fully and throws if it fails. `unsafe::cholesky(A)` skips the requirement, and the
/// name makes that easy to find later.
#include <cstdio>
#include <exception>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

namespace {

void print(const char *name, const vec<real> &x) {
    std::printf("%-26s[%g, %g, %g, %g]\n", name, x[0], x[1], x[2], x[3]);
}

} // namespace

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";

    // The 1D Laplacian, which is SPD. As a plain mat it carries no law.
    mat<real> A(4, 4, 0.0);
    for (idx i = 0; i < 4; ++i) {
        A(i, i) = 2.0;
        if (i + 1 < 4) {
            A(i, i + 1) = A(i + 1, i) = -1.0;
        }
    }
    const vec<real> b{1.0, 2.0, 2.0, 1.0};

    // auto F = cholesky(A);  // does not compile: A carries no SPD law

    const auto spd = assume_spd(A);
    static_assert(spd_operator<decltype(spd)>);
    static_assert(self_adjoint_operator<decltype(spd)>, "SPD implies self-adjoint");
    vec<real> x;
    solve(cholesky(spd), b, x);
    print("cholesky(assume_spd(A))", x);

    solve(cholesky(make_spd(A)), b, x);
    print("cholesky(make_spd(A))", x);

    solve(unsafe::cholesky(A), b, x);
    print("unsafe::cholesky(A)", x);

    mat<real> B(2, 2, 0.0);
    B(0, 0) = B(1, 1) = 1.0;
    B(0, 1) = B(1, 0) = 3.0; // eigenvalues 4 and -2
    try {
        (void)make_spd(B);
    } catch (const std::exception &e) {
        std::printf("make_spd([[1, 3], [3, 1]]) threw: %s\n", e.what());
    }

    if (plot) {
        std::vector<double> grid{0.0, 1.0, 2.0, 3.0}, values(x.data(), x.data() + 4);
        plt::plot(grid, values, "x", "linespoints");
        plt::title("14 Solution of the 1D Poisson problem");
        plt::show();
    }
}
