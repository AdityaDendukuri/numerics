/// @file 17_woodbury_low_rank_update.cpp
/// @brief Solve after a small matrix change without factoring from scratch.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numerics.hpp>

using namespace num;

namespace {

spmat to_sparse(const mat<real> &A) {
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            if (A(i, j) == 0.0)
                continue;
            rows.push_back(i);
            columns.push_back(j);
            values.push_back(A(i, j));
        }
    }
    return spmat::from_triplets(A.rows(), A.cols(), rows, columns, values);
}

} // namespace

int main() {
    constexpr idx n = 12;
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i + 1 < n)
            A(i, i + 1) = A(i + 1, i) = -1.0;
    }
    const auto original = lu(A);

    mat<real> changed = A;
    changed(5, 5) += 1.0;
    changed(5, 6) = changed(6, 5) = -0.5;
    const array<idx> changed_states{5, 6};
    const auto difference = low_rank_difference(to_sparse(A), to_sparse(changed), changed_states);
    const woodbury_solver<lu_result<real>> updated(original, difference);

    const vec<real> b(n, 1.0);
    const vec<real> reused = solve(updated, b);
    const vec<real> fresh = solve(lu(changed), b);
    real error = 0.0;
    for (idx i = 0; i < n; ++i)
        error = std::max(error, std::abs(reused[i] - fresh[i]));
    std::printf("rank %zu update, max error %.2e\n", static_cast<std::size_t>(updated.rank()),
                error);
}
