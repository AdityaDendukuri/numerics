/// @file tests/test_fields.cpp
/// @brief 3D field container + field_solver coverage (Phase 3 rewrite).
#include "fields/field3d.hpp"
#include "container/vector_ops.hpp"
#include "pde/field_solver.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace num;

// Values written through operator()/fill are visible through as_vec() at the
// grid's flat index -- i.e. the field's storage really is one num::vec.
TEST(scalar_field_3d, StorageRoundTrip) {
    scalar_field_3d f(4, 5, 6, 1.0f);
    EXPECT_EQ(f.size(), static_cast<idx>(4 * 5 * 6));

    f.fill([](int i, int j, int k) { return (100 * i) + (10 * j) + k; });

    for (int k = 0; k < f.nz(); ++k) {
        for (int j = 0; j < f.ny(); ++j) {
            for (int i = 0; i < f.nx(); ++i) {
                const double expected = (100 * i) + (10 * j) + k;
                EXPECT_DOUBLE_EQ(f(i, j, k), expected);
                EXPECT_DOUBLE_EQ(f.as_vec()[f.grid().flat(i, j, k)], expected);
            }
        }
    }
}

TEST(vector_field_3d, Scale) {
    vector_field_3d v(3, 3, 3, 1.0f);
    v.x.fill(1.0);
    v.y.fill(-2.0);
    v.z.fill(0.5);
    v.scale(2.0f);
    EXPECT_DOUBLE_EQ(v.x(1, 1, 1), 2.0);
    EXPECT_DOUBLE_EQ(v.y(1, 1, 1), -4.0);
    EXPECT_DOUBLE_EQ(v.z(1, 1, 1), 1.0);
}

// Manufactured solution: for any phi that is zero on the boundary, feeding the
// discrete Laplacian of phi as the source must recover phi. This exercises the
// in-place solve into phi.as_vec() (no to_vector/from_vector copy).
TEST(field_solver, PoissonManufacturedSolution) {
    const int n = 8;
    const float dx = 1.0f; // dx = 1 => operator works in pure index space.

    // Separable parabola product: zero on every boundary face, O(1) magnitude.
    auto para = [n](int m) {
        const double t = static_cast<double>(m) / (n - 1);
        return t * (1.0 - t);
    };
    auto phi_exact = [&](int i, int j, int k) { return para(i) * para(j) * para(k); };

    scalar_field_3d exact(n, n, n, dx, [&](int i, int j, int k) { return phi_exact(i, j, k); });

    // source = discrete Laplacian of phi_exact on the interior (dx^2 = 1).
    scalar_field_3d source(n, n, n, dx);
    for (int k = 1; k < n - 1; ++k) {
        for (int j = 1; j < n - 1; ++j) {
            for (int i = 1; i < n - 1; ++i) {
                const double lap = phi_exact(i + 1, j, k) + phi_exact(i - 1, j, k) +
                                   phi_exact(i, j + 1, k) + phi_exact(i, j - 1, k) +
                                   phi_exact(i, j, k + 1) + phi_exact(i, j, k - 1) -
                                   (6.0 * phi_exact(i, j, k));
                source(i, j, k) = lap;
            }
        }
    }

    scalar_field_3d phi(n, n, n, dx); // initial guess: 0
    const auto result = field_solver::solve_poisson(phi, source, 1e-9, 2000);
    ASSERT_TRUE(result.converged);

    double max_err = 0.0;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                max_err = std::max(max_err, std::abs(phi(i, j, k) - phi_exact(i, j, k)));
            }
        }
    }
    EXPECT_LT(max_err, 1e-6);

    // Boundary stays pinned to zero.
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(phi(i, j, 0), 0.0, 1e-9);
        }
    }
}

// Central-difference gradient of phi = x (i.e. i*dx) is exactly 1 in x, 0 else.
TEST(field_solver, GradientOfLinearFieldIsConstant) {
    const int n = 6;
    const float dx = 2.0f;
    scalar_field_3d phi(n, n, n, dx, [dx](int i, int, int) { return i * static_cast<double>(dx); });

    const vector_field_3d g = field_solver::gradient(phi);
    EXPECT_NEAR(g.x(2, 2, 2), 1.0, 1e-9);
    EXPECT_NEAR(g.y(2, 2, 2), 0.0, 1e-9);
    EXPECT_NEAR(g.z(2, 2, 2), 0.0, 1e-9);
}
