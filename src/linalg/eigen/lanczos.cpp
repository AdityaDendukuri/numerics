/// @file eigen/lanczos.cpp
/// @brief Lanczos algorithm with full reorthogonalisation
///
/// The k-step Lanczos recurrence:
///
///   beta_0 = 0,  v_0 = 0
///   v_1 = random unit vector
///   for j = 1..k:
///     w  = A*v_j
///     alpha_j = v_j^T * w
///     w  = w - alpha_j*v_j - beta_{j-1}*v_{j-1}
///     [reorthogonalise w against all previous v_1..v_j]
///     beta_j = ||w||
///     v_{j+1} = w / beta_j
///
/// This builds the symmetric tridiagonal T_k = V_k^T * A * V_k with diagonal alpha
/// and off-diagonal beta.  The Ritz values are the eigenvalues of T_k and the
/// Ritz vectors are V_k * (eigenvectors of T_k).
///
/// T_k is diagonalised with eig_sym (Jacobi), which is O(k^3)  -- cheap for k << n.

#include "linalg/eigen/lanczos.hpp"
#include "linalg/eigen/jacobi_eig.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace num {

LanczosResult lanczos(MatVecFn matvec, idx n, idx k,
                      real tol, idx max_steps, Backend /*backend*/) {
    if (k == 0 || k > n)
        throw std::invalid_argument("lanczos: k must satisfy 0 < k <= n");

    if (max_steps == 0) max_steps = std::min(3*k, n);
    max_steps = std::min(max_steps, n);

    // Lanczos basis: columns of V (n x max_steps)
    // Store as a flat array for efficient column access
    Matrix V(n, max_steps, 0.0);

    // alpha (diagonal) and beta (off-diagonal) of the tridiagonal T
    Vector alpha(max_steps, 0.0);
    Vector beta(max_steps, 0.0);   // beta[j] = beta_{j+1}, beta[0] unused

    // Initialise v_1 = e_1 (deterministic, avoids Rng dependency)
    for (idx i = 0; i < n; ++i) V(i, 0) = (i == 0) ? 1.0 : 0.0;

    idx steps = 0;

    for (idx j = 0; j < max_steps; ++j) {
        // Extract current Lanczos vector v_j from column j of V
        Vector vj(n);
        for (idx i = 0; i < n; ++i) vj[i] = V(i, j);

        // w = A * vj
        Vector w(n, 0.0);
        matvec(vj, w);

        // alpha_j = v_j^T * w
        real a = 0;
        for (idx i = 0; i < n; ++i) a += vj[i] * w[i];
        alpha[j] = a;

        // w = w - alpha_j*v_j - beta_{j-1}*v_{j-1}
        for (idx i = 0; i < n; ++i) w[i] -= a * vj[i];
        if (j > 0) {
            for (idx i = 0; i < n; ++i) w[i] -= beta[j-1] * V(i, j-1);
        }

        // Full reorthogonalisation (modified Gram-Schmidt against all previous)
        for (idx l = 0; l <= j; ++l) {
            real proj = 0;
            for (idx i = 0; i < n; ++i) proj += V(i, l) * w[i];
            for (idx i = 0; i < n; ++i) w[i] -= proj * V(i, l);
        }

        // beta_j = ||w||
        real b = 0;
        for (idx i = 0; i < n; ++i) b += w[i]*w[i];
        b = std::sqrt(b);

        ++steps;

        // Invariant subspace found  -- can't continue
        if (b < 1e-12) break;

        beta[j] = b;

        // Store v_{j+1} if room remains
        if (j + 1 < max_steps) {
            for (idx i = 0; i < n; ++i) V(i, j+1) = w[i] / b;
        }
    }

    // Build steps x steps tridiagonal T from (alpha, beta) and diagonalise it.
    // Using all 'steps' Lanczos vectors (not just k) gives far better Ritz approximations.
    idx m = steps;
    Matrix T(m, m, 0.0);
    for (idx j = 0; j < m; ++j) {
        T(j, j) = alpha[j];
        if (j + 1 < m) {
            T(j, j+1) = beta[j];
            T(j+1, j) = beta[j];
        }
    }

    EigenResult teig = eig_sym(T, tol * 1e-2);  // tighter tol for inner solve
    // teig.values are in ascending order; the top-k are at indices m-k .. m-1

    idx nret = std::min(k, m);   // how many Ritz pairs to return

    // Ritz vectors for the top-nret eigenvalues: V_m * eigvecs_of_T[:, m-nret..m-1]
    Matrix ritz_vecs(n, nret, 0.0);
    for (idx i = 0; i < nret; ++i) {
        idx ti = m - nret + i;   // column in teig (ascending order; largest last)
        for (idx j = 0; j < m; ++j) {
            real coeff = teig.vectors(j, ti);
            for (idx r = 0; r < n; ++r)
                ritz_vecs(r, i) += coeff * V(r, j);
        }
    }

    // Slice the top-nret eigenvalues (ascending)
    Vector ritz_vals(nret);
    for (idx i = 0; i < nret; ++i)
        ritz_vals[i] = teig.values[m - nret + i];

    // Check convergence of each Ritz pair: ||A*u - lambda*u||
    bool all_converged = true;
    for (idx i = 0; i < nret; ++i) {
        Vector u(n);
        for (idx r = 0; r < n; ++r) u[r] = ritz_vecs(r, i);

        Vector Au(n, 0.0);
        matvec(u, Au);

        real res = 0;
        real lam = ritz_vals[i];
        for (idx r = 0; r < n; ++r) {
            real d = Au[r] - lam * u[r];
            res += d*d;
        }
        if (std::sqrt(res) > tol) { all_converged = false; break; }
    }

    return {ritz_vals, ritz_vecs, steps, all_converged};
}

} // namespace num
