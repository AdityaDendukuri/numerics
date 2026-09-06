/// @file benchmarks/bench_convergence.cpp
/// @brief Generates convergence traces, solver comparisons, and diagnostic plots for the report.
#include <cmath>
#include "container/vector_ops.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <numerics.hpp>

namespace fs = std::filesystem;

namespace {

/// Generate a variable-coefficient 2D Laplacian matrix with varying diagonal entries: -\div(a(x,y)\grad u).
num::mat make_variable_laplacian_2d(num::idx n_side) {
    const num::idx n = n_side * n_side;
    num::mat A(n, n, 0.0);
    const auto id = [n_side](num::idx x, num::idx y) { return y * n_side + x; };

    for (num::idx y = 0; y < n_side; ++y) {
        for (num::idx x = 0; x < n_side; ++x) {
            num::idx u = id(x, y);
            double ax_plus = 1.0 + 20.0 * std::pow(std::sin(3.14159 * (x + 0.5) / n_side), 2);
            double ax_minus = 1.0 + 20.0 * std::pow(std::sin(3.14159 * (x - 0.5) / n_side), 2);
            double ay_plus = 1.0 + 20.0 * std::pow(std::cos(3.14159 * (y + 0.5) / n_side), 2);
            double ay_minus = 1.0 + 20.0 * std::pow(std::cos(3.14159 * (y - 0.5) / n_side), 2);

            double diag = 0.0;
            if (x + 1 < n_side) { A(u, id(x + 1, y)) = -ax_plus; diag += ax_plus; }
            if (x > 0)          { A(u, id(x - 1, y)) = -ax_minus; diag += ax_minus; }
            if (y + 1 < n_side) { A(u, id(x, y + 1)) = -ay_plus; diag += ay_plus; }
            if (y > 0)          { A(u, id(x, y - 1)) = -ay_minus; diag += ay_minus; }
            A(u, u) = diag + 0.1;
        }
    }
    return A;
}

/// Run iterative solvers and record residual histories ||r_k|| / ||b||.
void generate_iterative_convergence_plot(const std::string &out_dir) {
    using namespace num;
    const idx n_side = 32;
    const idx n = n_side * n_side;

    mat A = make_variable_laplacian_2d(n_side);
    vec b(n, 0.0);
    const auto id = [n_side](idx x, idx y) { return y * n_side + x; };
    for (idx y = 0; y < n_side; ++y) {
        for (idx x = 0; x < n_side; ++x) {
            b[id(x, y)] = std::sin(4.0 * 3.14159 * x / n_side) * std::cos(4.0 * 3.14159 * y / n_side);
        }
    }
    const double b_norm = norm(b);

    operators::dense_op dense_op{A};
    operators::spd_op<operators::dense_op> spd_op{dense_op};
    auto jacobi = make_jacobi_preconditioner(A);

    const idx max_iters = 80;

    // 1. Trace Conjugate Gradient
    std::vector<double> iters_cg, res_cg;
    {
        iters_cg.push_back(0.0);
        res_cg.push_back(1.0);
        for (idx k = 1; k <= max_iters; ++k) {
            vec x(n, 0.0);
            auto res = cg(spd_op, b, x, 1e-15, k);
            iters_cg.push_back(static_cast<double>(k));
            res_cg.push_back(std::max(1e-16, res.residual / b_norm));
            if (res.residual / b_norm < 1e-15) break;
        }
    }

    // 2. Trace Preconditioned CG (Jacobi)
    std::vector<double> iters_pcg, res_pcg;
    {
        iters_pcg.push_back(0.0);
        res_pcg.push_back(1.0);
        for (idx k = 1; k <= max_iters; ++k) {
            vec x(n, 0.0);
            auto res = pcg(spd_op, jacobi, b, x, 1e-15, k);
            iters_pcg.push_back(static_cast<double>(k));
            res_pcg.push_back(std::max(1e-16, res.residual / b_norm));
            if (res.residual / b_norm < 1e-15) break;
        }
    }

    // 3. Trace GMRES (Restarted m=30) with corrected parameter order
    std::vector<double> iters_gmres, res_gmres;
    {
        iters_gmres.push_back(0.0);
        res_gmres.push_back(1.0);
        for (idx k = 1; k <= max_iters; ++k) {
            vec x(n, 0.0);
            auto res = gmres(dense_op, b, x, 1e-15, /*max_iter=*/k, /*restart=*/30);
            iters_gmres.push_back(static_cast<double>(k));
            res_gmres.push_back(std::max(1e-16, res.residual / b_norm));
            if (res.residual / b_norm < 1e-15) break;
        }
    }

    // 4. Trace MINRES
    std::vector<double> iters_minres, res_minres;
    {
        iters_minres.push_back(0.0);
        res_minres.push_back(1.0);
        for (idx k = 1; k <= max_iters; ++k) {
            vec x(n, 0.0);
            auto res = minres(spd_op, b, x, 1e-15, k);
            iters_minres.push_back(static_cast<double>(k));
            res_minres.push_back(std::max(1e-16, res.residual / b_norm));
            if (res.residual / b_norm < 1e-15) break;
        }
    }

    // 5. Trace Gauss-Seidel
    std::vector<double> iters_gs, res_gs;
    {
        vec x(n, 0.0);
        iters_gs.push_back(0.0);
        res_gs.push_back(1.0);
        for (idx k = 1; k <= max_iters; ++k) {
            for (idx i = 0; i < n; ++i) {
                double sigma = 0.0;
                for (idx j = 0; j < n; ++j) {
                    if (j != i) sigma += A(i, j) * x[j];
                }
                x[i] = (b[i] - sigma) / A(i, i);
            }
            vec r(n, 0.0);
            dense_op.apply(x, r);
            axpy(-1.0, b, r);
            iters_gs.push_back(static_cast<double>(k));
            res_gs.push_back(std::max(1e-16, norm(r) / b_norm));
        }
    }

    // 6. Trace Jacobi
    std::vector<double> iters_jac, res_jac;
    {
        vec x(n, 0.0);
        iters_jac.push_back(0.0);
        res_jac.push_back(1.0);
        for (idx k = 1; k <= max_iters; ++k) {
            vec x_new(n, 0.0);
            for (idx i = 0; i < n; ++i) {
                double sigma = 0.0;
                for (idx j = 0; j < n; ++j) {
                    if (j != i) sigma += A(i, j) * x[j];
                }
                x_new[i] = (b[i] - sigma) / A(i, i);
            }
            x = x_new;
            vec r(n, 0.0);
            dense_op.apply(x, r);
            axpy(-1.0, b, r);
            iters_jac.push_back(static_cast<double>(k));
            res_jac.push_back(std::max(1e-16, norm(r) / b_norm));
        }
    }

    // Plot residual curves
    plt::plot(iters_cg, res_cg, "Conjugate Gradient (cg_method)", "lines lw 2.5 lc rgb '#1f77b4'");
    plt::plot(iters_pcg, res_pcg, "Preconditioned cg_method (Jacobi)", "lines lw 2.5 lc rgb '#2ca02c'");
    plt::plot(iters_gmres, res_gmres, "gmres_method (m=30)", "lines lw 2.5 lc rgb '#d62728'");
    plt::plot(iters_minres, res_minres, "minres_method", "lines lw 2.0 lc rgb '#9467bd'");
    plt::plot(iters_gs, res_gs, "Gauss-Seidel", "lines lw 2.0 lc rgb '#ff7f0e'");
    plt::plot(iters_jac, res_jac, "Jacobi Iteration", "lines lw 1.8 lc rgb '#8c564b'");

    plt::title("Iterative Solvers Convergence (Variable-Coefficient 2D Laplacian, N=1024)");
    plt::xlabel("Iteration (k)");
    plt::ylabel("Relative Residual ||r_k|| / ||b||");
    plt::semilogy();
    plt::legend("top right");

    std::string plot_path = out_dir + "/iterative_convergence.png";
    plt::savefig(plot_path);
    std::cout << "Saved: " << plot_path << "\n";
}

/// Generate Wikipedia-style comparison of MINRES vs CG (Error and Residual Norms).
void generate_cg_vs_minres_plot(const std::string &out_dir) {
    using namespace num;
    const idx n = 100;

    // Generate well-conditioned SPD test matrix with known exact solution
    mat A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = 2.0 + 0.5 * (i + 1);
        if (i > 0) {
            A(i, i - 1) = -1.0;
            A(i - 1, i) = -1.0;
        }
    }

    vec x_star(n, 1.0);
    vec b(n, 0.0);
    operators::dense_op dense_op{A};
    dense_op.apply(x_star, b);

    operators::spd_op<operators::dense_op> spd_op{dense_op};

    std::vector<double> iters;
    std::vector<double> err_minres, res_minres, err_cg, res_cg;

    const idx max_steps = 25;

    for (idx k = 1; k <= max_steps; ++k) {
        iters.push_back(static_cast<double>(k));

        // MINRES
        vec x_m(n, 0.0);
        auto res_m = minres(spd_op, b, x_m, 1e-15, k);
        vec e_m(n, 0.0);
        for (idx i = 0; i < n; ++i) e_m[i] = x_m[i] - x_star[i];
        err_minres.push_back(std::max(1e-16, norm(e_m)));
        res_minres.push_back(std::max(1e-16, res_m.residual));

        // CG
        vec x_c(n, 0.0);
        auto res_c = cg(spd_op, b, x_c, 1e-15, k);
        vec e_c(n, 0.0);
        for (idx i = 0; i < n; ++i) e_c[i] = x_c[i] - x_star[i];
        err_cg.push_back(std::max(1e-16, norm(e_c)));
        res_cg.push_back(std::max(1e-16, res_c.residual));
    }

    plt::plot(iters, err_minres, "Error ||x_k - x*|| (minres_method)", "lines lw 2.5 lc rgb '#2ca02c'");
    plt::plot(iters, res_minres, "Residual ||r_k|| (minres_method)", "lines dt 2 lw 2.5 lc rgb '#2ca02c'");
    plt::plot(iters, err_cg, "Error ||x_k - x*|| (cg_method)", "lines lw 2.5 lc rgb '#1f77b4'");
    plt::plot(iters, res_cg, "Residual ||r_k|| (cg_method)", "lines dt 2 lw 2.5 lc rgb '#1f77b4'");

    plt::title("minres_method vs Conjugate Gradient (Error and Residual Norms)");
    plt::xlabel("Iteration (k)");
    plt::ylabel("Error / Residual Euclidean Norm");
    plt::semilogy();
    plt::legend("top right");

    std::string plot_path = out_dir + "/cg_vs_minres.png";
    plt::savefig(plot_path);
    std::cout << "Saved: " << plot_path << "\n";
}

/// Generate Talbot contour exponential convergence plot.
void generate_talbot_convergence_plot(const std::string &out_dir) {
    using namespace num;
    const idx N = 100;
    std::mt19937_64 rng(42);

    graph G = structures::erdos_renyi(N, 0.08, rng, true, 0.5, 2.0);
    mat Q = num::linear::dense_markov_generator(G, true);
    vec p0 = unit_vector(N, 0);

    const double t = 1.0;
    operators::dense_op Q_op(Q);
    vec p_exact = expv(t, Q_op, p0, 50, 1e-15);

    hessenberg_resolvent_solver solver(Q);

    const std::vector<idx> node_counts = {4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32};
    std::vector<double> nodes_dbl;
    std::vector<double> errors_inf;
    std::vector<double> errors_l1;

    for (idx M : node_counts) {
        std::vector<cplx> density(N, cplx(0.0, 0.0));
        inverse_laplace_accumulate(t, M, [&](cplx shift, cplx weight) {
            auto sol = solver.solve(shift, p0);
            for (idx i = 0; i < N; ++i) density[i] += weight * sol[i];
        });

        vec p_talbot(N, 0.0);
        for (idx i = 0; i < N; ++i) p_talbot[i] = std::max(0.0, density[i].real());
        clip_and_normalize_nonnegative(p_talbot);

        double l_inf = 0.0;
        double l_1 = 0.0;
        for (idx i = 0; i < N; ++i) {
            double diff = std::abs(p_talbot[i] - p_exact[i]);
            l_inf = std::max(l_inf, diff);
            l_1 += diff;
        }

        nodes_dbl.push_back(static_cast<double>(M));
        errors_inf.push_back(std::max(1e-16, l_inf));
        errors_l1.push_back(std::max(1e-16, l_1));
    }

    plt::plot(nodes_dbl, errors_inf, "L_inf Error", "linespoints pt 7 ps 1.2 lw 2.5 lc rgb '#1f77b4'");
    plt::plot(nodes_dbl, errors_l1, "L_1 Error", "linespoints pt 5 ps 1.2 lw 2.5 lc rgb '#ff7f0e'");
    plt::title("Talbot Inverse Laplace Spectral Convergence (t=1.0, N=100)");
    plt::xlabel("Quadrature Nodes (M)");
    plt::ylabel("Absolute Error vs Exact Arnoldi Action");
    plt::semilogy();
    plt::legend("top right");

    std::string plot_path = out_dir + "/talbot_convergence.png";
    plt::savefig(plot_path);
    std::cout << "Saved: " << plot_path << "\n";
}

/// Generate Symplectic Hamiltonian Energy Preservation Plot.
void generate_symplectic_energy_plot(const std::string &out_dir) {
    using namespace num;

    // Harmonic oscillator H(q, p) = 0.5*p^2 + 0.5*q^2
    const double h = 0.05;
    const idx n_steps = 2000;

    const auto energy = [](double q, double p) { return 0.5 * p * p + 0.5 * q * q; };
    const double E0 = energy(1.0, 0.0);

    std::vector<double> time_steps;
    std::vector<double> e_euler, e_rk4, e_verlet, e_yoshida;

    // 1. Explicit Euler
    {
        double q = 1.0, p = 0.0;
        for (idx step = 0; step <= n_steps; ++step) {
            double t = step * h;
            if (step % 10 == 0) {
                time_steps.push_back(t);
                e_euler.push_back(std::max(1e-16, std::abs(energy(q, p) - E0)));
            }
            double q_next = q + h * p;
            double p_next = p - h * q;
            q = q_next;
            p = p_next;
        }
    }

    // 2. Classical RK4
    {
        double q = 1.0, p = 0.0;
        for (idx step = 0; step <= n_steps; ++step) {
            if (step % 10 == 0) {
                e_rk4.push_back(std::max(1e-16, std::abs(energy(q, p) - E0)));
            }
            // f(q, p) = (p, -q)
            double kq1 = p;
            double kp1 = -q;

            double kq2 = p + 0.5 * h * kp1;
            double kp2 = -(q + 0.5 * h * kq1);

            double kq3 = p + 0.5 * h * kp2;
            double kp3 = -(q + 0.5 * h * kq2);

            double kq4 = p + h * kp3;
            double kp4 = -(q + h * kq3);

            q += (h / 6.0) * (kq1 + 2.0 * kq2 + 2.0 * kq3 + kq4);
            p += (h / 6.0) * (kp1 + 2.0 * kp2 + 2.0 * kp3 + kp4);
        }
    }

    // 3. Störmer-Verlet (2nd order symplectic)
    {
        double q = 1.0, p = 0.0;
        for (idx step = 0; step <= n_steps; ++step) {
            if (step % 10 == 0) {
                e_verlet.push_back(std::max(1e-16, std::abs(energy(q, p) - E0)));
            }
            // Kick-Drift-Kick
            p -= 0.5 * h * q;
            q += h * p;
            p -= 0.5 * h * q;
        }
    }

    // 4. Yoshida (4th order symplectic)
    {
        const double w0 = -std::cbrt(2.0) / (2.0 - std::cbrt(2.0));
        const double w1 = 1.0 / (2.0 - std::cbrt(2.0));
        const double c1 = w1 / 2.0;
        const double c2 = (w0 + w1) / 2.0;
        const double c3 = c2;
        const double c4 = c1;
        const double d1 = w1;
        const double d2 = w0;
        const double d3 = w1;

        double q = 1.0, p = 0.0;
        for (idx step = 0; step <= n_steps; ++step) {
            if (step % 10 == 0) {
                e_yoshida.push_back(std::max(1e-16, std::abs(energy(q, p) - E0)));
            }
            q += c1 * h * p;
            p -= d1 * h * q;
            q += c2 * h * p;
            p -= d2 * h * q;
            q += c3 * h * p;
            p -= d3 * h * q;
            q += c4 * h * p;
        }
    }

    plt::plot(time_steps, e_euler, "Explicit euler_method (O(t) Explosion)", "lines lw 2.0 lc rgb '#d62728'");
    plt::plot(time_steps, e_rk4, "Classical rk4_method (Dissipative Drift)", "lines lw 2.0 lc rgb '#ff7f0e'");
    plt::plot(time_steps, e_verlet, "Störmer-Verlet 2nd-Order (Bounded O(h^2))", "lines lw 2.0 lc rgb '#1f77b4'");
    plt::plot(time_steps, e_yoshida, "Yoshida 4th-Order Symplectic (Bounded O(h^4))", "lines lw 2.5 lc rgb '#2ca02c'");

    plt::title("Hamiltonian Energy Error Conservation |E(t) - E(0)| (h=0.05)");
    plt::xlabel("Time t");
    plt::ylabel("Absolute Energy Drift |E(t) - E0|");
    plt::semilogy();
    plt::legend("top left");

    std::string plot_path = out_dir + "/symplectic_energy.png";
    plt::savefig(plot_path);
    std::cout << "Saved: " << plot_path << "\n";
}

} // namespace

int main(int argc, char **argv) {
    std::string out_dir = (argc > 1) ? argv[1] : "output/plots";
    fs::create_directories(out_dir);

    std::cout << "Generating convergence and dynamical diagnostics in " << out_dir << "/ ...\n";
    generate_iterative_convergence_plot(out_dir);
    generate_cg_vs_minres_plot(out_dir);
    generate_talbot_convergence_plot(out_dir);
    generate_symplectic_energy_plot(out_dir);
    std::cout << "All diagnostic plots generated successfully.\n";
    return 0;
}
