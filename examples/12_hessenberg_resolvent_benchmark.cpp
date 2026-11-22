/// @file 12_hessenberg_resolvent_benchmark.cpp
/// @brief Timing shifted solves with one Hessenberg reduction against one per shift.
///
/// Solving \f$(sI - A)x = b\f$ at k shifts costs \f$O(kn^3)\f$ if A is reduced to Hessenberg form
/// at every shift, and \f$O(n^3 + kn^2)\f$ if it is reduced once. Both use the same arithmetic per
/// shift, so the solutions agree.
#include <chrono>
#include <cstdio>
#include <numerics.hpp>
#include <random>
#include <vector>

using namespace num;

namespace {

template <class Work>
double milliseconds(Work &&work) {
    const auto start = std::chrono::steady_clock::now();
    work();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> entry(-1.0, 1.0);

    // 100 shifts along the line Re s = 0.1, as in a frequency sweep.
    array<cplx> shifts;
    for (int k = 0; k < 100; ++k) {
        shifts.push_back(cplx(0.1, 0.1 * k));
    }

    std::printf("   n  shifts  reduce per shift (ms)  reduce once (ms)  speedup  max difference\n");
    for (idx n : {50, 100, 200, 400}) {
        mat<real> A(n, n, 0.0);
        for (idx i = 0; i < n; ++i) {
            for (idx j = 0; j < n; ++j) {
                A(i, j) = entry(rng);
            }
            A(i, i) += 5.0;
        }
        vec<real> b(n);
        for (idx i = 0; i < n; ++i) {
            b[i] = entry(rng);
        }

        std::vector<vec<cplx>> each(shifts.size());
        const double each_ms = milliseconds([&] {
            for (idx k = 0; k < shifts.size(); ++k) {
                const hessenberg_resolvent R(A);
                each[k] = solve(shift(R, shifts[k]), b);
            }
        });

        array<vec<cplx>> once;
        const double once_ms = milliseconds([&] {
            const hessenberg_resolvent R(A);
            once = solve_batch(R, shifts, b);
        });

        real difference = 0.0;
        for (idx k = 0; k < shifts.size(); ++k) {
            for (idx i = 0; i < n; ++i) {
                difference = std::max(difference, std::abs(each[k][i] - once[k][i]));
            }
        }
        std::printf("%4zu  %6zu  %21.2f  %16.2f  %6.1fx  %14.1e\n", static_cast<std::size_t>(n),
                    static_cast<std::size_t>(shifts.size()), each_ms, once_ms, each_ms / once_ms,
                    difference);
    }
}
