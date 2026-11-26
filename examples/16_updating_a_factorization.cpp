/// @file 16_updating_a_factorization.cpp
/// @brief Update a factorization after a few rows and columns of the matrix change.
#include <algorithm>
#include <cmath>
#include <io/json.hpp>
#include <numerics.hpp>

using namespace num;

namespace {

spmat chain(const array<idx> &labels) {
    const idx n = labels.size();
    array<idx> rows, columns;
    array<real> values;
    auto add = [&](idx row, idx column, real value) {
        rows.push_back(row);
        columns.push_back(column);
        values.push_back(value);
    };
    for (idx i = 0; i < n; ++i) {
        const real rate = 0.5 + 0.1 * static_cast<real>(labels[i] % 5);
        real diagonal = 0.2;
        if (i > 0) {
            add(i, i - 1, -rate);
            diagonal += rate;
        }
        if (i + 1 < n) {
            add(i, i + 1, -rate);
            diagonal += rate;
        }
        add(i, i, diagonal);
    }
    return spmat::from_triplets(n, n, rows, columns, values);
}

template <class Factor>
real error_against_fresh(const Factor &factor, const spmat &A) {
    const vec<real> b(A.n_rows(), 1.0);
    const vec<real> reused = solve(factor, b);
    const vec<real> fresh = solve(lu(dense(A)), b);
    real error = 0.0;
    for (idx i = 0; i < b.size(); ++i)
        error = std::max(error, std::abs(reused[i] - fresh[i]));
    return error;
}

} // namespace

int main() {
    constexpr idx n = 20;
    array<idx> labels(n), levels(n);
    for (idx i = 0; i < n; ++i) {
        labels[i] = i;
        levels[i] = i / 4;
    }

    corrected_lu corrected = lu(chain(labels), no_pivot, nullptr, {});
    labels[7] += 1;
    const spmat changed = chain(labels);
    corrected = lu(changed, no_pivot, &corrected, array<idx>{7});
    const real corrected_error = error_against_fresh(corrected, changed);

    // Changing block 3 keeps blocks 0 through 2 and refactors blocks 3 and 4.
    suffix_block_lu blocked = lu(changed, blocks(levels), nullptr, {});
    labels[15] += 1;
    const spmat changed_again = chain(labels);
    blocked = lu(changed_again, blocks(levels), &blocked, array<idx>{15});
    io::write_json({
        {"corrected_lu", {{"reused", corrected.reused()}, {"max_error", corrected_error}}},
        {"block_lu", {{"reused_prefix_blocks", blocked.reused_blocks},
                      {"refactored_suffix_blocks", n / 4 - blocked.reused_blocks},
                      {"max_error", error_against_fresh(blocked, changed_again)}}}
    }, "result.json");
}
