/// @file 16_factor_reuse.cpp
/// @brief Refactoring after local changes, with Woodbury-corrected LU and prefix-reusing block LU.
///
/// A sweep over a Markov chain changes a few states at a time. Each change touches one row and one
/// column of the rate matrix R, so the previous factorization is nearly right. `lu(R, no_pivot,
/// &previous, changed)` corrects it with a Woodbury update while at most three states differ from
/// the last full factorization, and `lu(R, blocks(levels), &previous, changed)` keeps every block
/// before the first change.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numerics.hpp>

using namespace num;

namespace {

// A chain of n states. State i jumps to its neighbours at rates set by the labels of the two
// states, and decays at rate 0.2, so R = diag(total rate) - (jump rates) is a nonsingular
// M-matrix. Changing a label changes that state's row and column.
spmat chain(const array<idx> &label) {
    const idx n = label.size();
    auto rate = [&](idx from, idx to) {
        return 0.5 + (0.1 * static_cast<real>(((label[from] * 7) + (label[to] * 3)) % 5));
    };
    array<idx> rows, columns;
    array<real> values;
    auto entry = [&](idx i, idx j, real value) {
        rows.push_back(i);
        columns.push_back(j);
        values.push_back(value);
    };
    for (idx i = 0; i < n; ++i) {
        real total = 0.2;
        if (i > 0) {
            entry(i, i - 1, -rate(i, i - 1));
            total += rate(i, i - 1);
        }
        if (i + 1 < n) {
            entry(i, i + 1, -rate(i, i + 1));
            total += rate(i, i + 1);
        }
        entry(i, i, total);
    }
    return spmat::from_triplets(n, n, rows, columns, values);
}

// Largest difference from a fresh dense LU of R.
template <class F>
real error_against_fresh(const F &factor, const spmat &R) {
    const vec<real> b(R.n_rows(), 1.0);
    const vec<real> x = solve(factor, b);
    const vec<real> expected = solve(lu(dense(R)), b);
    real worst = 0.0;
    for (idx i = 0; i < x.size(); ++i) {
        worst = std::max(worst, std::abs(x[i] - expected[i]));
    }
    return worst;
}

template <class Work>
double milliseconds(Work &&work) {
    const auto start = std::chrono::steady_clock::now();
    work();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

int main() {
    constexpr idx n = 600;
    array<idx> label(n), levels(n);
    for (idx i = 0; i < n; ++i) {
        label[i] = i;
        levels[i] = i / 4; // blocks of four states; the chain couples only adjacent blocks
    }
    std::cout << std::scientific << std::setprecision(1);

    // 1. Dense no-pivot LU, corrected by Woodbury while few states differ.
    std::cout << "corrected no-pivot LU, n = " << n << "\n";
    corrected_lu Z = lu(chain(label), no_pivot, nullptr, {});
    for (idx slot : {37, 210, 37, 455, 512}) {
        label[slot] += 100;
        const spmat R = chain(label);
        const double fresh_ms = milliseconds([&] { (void)lu(R, no_pivot); });
        const double reuse_ms = milliseconds([&] { Z = lu(R, no_pivot, &Z, array<idx>{slot}); });
        std::cout << "  state " << std::setw(3) << slot << ": "
                  << (Z.reused() ? "corrected " : "refactored") << std::fixed
                  << std::setprecision(2) << "  " << reuse_ms << " ms (fresh " << fresh_ms
                  << " ms)" << std::scientific << std::setprecision(1)
                  << "  error " << error_against_fresh(Z, R) << "\n";
    }

    // 2. Block-tridiagonal LU, keeping every block before the first change.
    std::cout << "block LU, " << n / 4 << " blocks\n";
    suffix_block_lu B = lu(chain(label), blocks(levels), nullptr, {});
    for (idx slot : {590, 300, 3}) {
        label[slot] += 100;
        const spmat R = chain(label);
        B = lu(R, blocks(levels), &B, array<idx>{slot});
        std::cout << "  state " << std::setw(3) << slot << ": kept " << std::setw(3)
                  << B.reused_blocks << " of " << n / 4 << " blocks  error "
                  << error_against_fresh(B, R) << "\n";
    }
}
