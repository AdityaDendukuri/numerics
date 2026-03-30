#include "quantum/statevector.hpp"
#include <cmath>

namespace num {
namespace quantum {

using cplx = std::complex<real>;
static constexpr real pi = 3.141592653589793238462643383279502884;

// State construction

Statevector zero_state(int n_qubits) {
    Statevector sv(1 << n_qubits, cplx{0, 0});
    sv[0] = cplx{1, 0};
    return sv;
}

Statevector basis_state(int n_qubits, int k) {
    Statevector sv(1 << n_qubits, cplx{0, 0});
    sv[k] = cplx{1, 0};
    return sv;
}

// Single-qubit gates

Gate gate_x() { return {cplx{0,0}, cplx{1,0}, cplx{1,0}, cplx{0,0}}; }
Gate gate_y() { return {cplx{0,0}, cplx{0,-1}, cplx{0,1}, cplx{0,0}}; }
Gate gate_z() { return {cplx{1,0}, cplx{0,0}, cplx{0,0}, cplx{-1,0}}; }

Gate gate_h() {
    const real s = 1.0 / std::sqrt(real(2));
    return {cplx{s,0}, cplx{s,0}, cplx{s,0}, cplx{-s,0}};
}

Gate gate_s()   { return {cplx{1,0}, cplx{0,0}, cplx{0,0}, cplx{0,1}}; }
Gate gate_sdg() { return {cplx{1,0}, cplx{0,0}, cplx{0,0}, cplx{0,-1}}; }

Gate gate_t() {
    const real c = std::cos(pi / 4), s = std::sin(pi / 4);
    return {cplx{1,0}, cplx{0,0}, cplx{0,0}, cplx{c, s}};
}
Gate gate_tdg() {
    const real c = std::cos(pi / 4), s = std::sin(pi / 4);
    return {cplx{1,0}, cplx{0,0}, cplx{0,0}, cplx{c, -s}};
}

Gate gate_rx(real theta) {
    const real c = std::cos(theta / 2), s = std::sin(theta / 2);
    return {cplx{c,0}, cplx{0,-s}, cplx{0,-s}, cplx{c,0}};
}
Gate gate_ry(real theta) {
    const real c = std::cos(theta / 2), s = std::sin(theta / 2);
    return {cplx{c,0}, cplx{-s,0}, cplx{s,0}, cplx{c,0}};
}
Gate gate_rz(real theta) {
    const real c = std::cos(theta / 2), s = std::sin(theta / 2);
    return {cplx{c,-s}, cplx{0,0}, cplx{0,0}, cplx{c,s}};
}

Gate gate_p(real lambda) {
    return {cplx{1,0}, cplx{0,0}, cplx{0,0},
            cplx{std::cos(lambda), std::sin(lambda)}};
}

Gate gate_u(real theta, real phi, real lambda) {
    const real ct = std::cos(theta / 2), st = std::sin(theta / 2);
    return {
        cplx{ct, 0},
        cplx{-std::cos(lambda) * st, -std::sin(lambda) * st},
        cplx{ std::cos(phi)    * st,  std::sin(phi)    * st},
        cplx{ std::cos(phi + lambda) * ct, std::sin(phi + lambda) * ct}
    };
}

// Two-qubit gate matrices
// Basis ordering: |00>, |01>, |10>, |11> (little-endian: q0=LSB, q1=MSB)

Gate2Q gate2q_cnot() {
    Gate2Q G{}; // zero-init
    G[0]  = cplx{1,0}; // |00> -> |00>
    G[5]  = cplx{1,0}; // |01> -> |01>
    G[14] = cplx{1,0}; // |10> -> |11>  (ctrl=q1=1, flip q0)
    G[11] = cplx{1,0}; // |11> -> |10>
    return G;
}

Gate2Q gate2q_cz() {
    Gate2Q G{};
    G[0]  = cplx{1,0};
    G[5]  = cplx{1,0};
    G[10] = cplx{1,0};
    G[15] = cplx{-1,0}; // |11> -> -|11>
    return G;
}

Gate2Q gate2q_swap() {
    Gate2Q G{};
    G[0]  = cplx{1,0};  // |00> -> |00>
    G[9]  = cplx{1,0};  // |01> -> |10>
    G[6]  = cplx{1,0};  // |10> -> |01>
    G[15] = cplx{1,0};  // |11> -> |11>
    return G;
}

// Gate application

void apply_1q(Statevector& sv, int n_qubits, const Gate& G, int target) {
    const int stride = 1 << target;
    const int N = 1 << n_qubits;
    for (int i = 0; i < N; i += stride << 1) {
        for (int j = 0; j < stride; ++j) {
            const int i0 = i + j, i1 = i0 + stride;
            const cplx a0 = sv[i0], a1 = sv[i1];
            sv[i0] = G[0] * a0 + G[1] * a1;
            sv[i1] = G[2] * a0 + G[3] * a1;
        }
    }
}

// Apply a 4x4 gate to qubits q0 (LSB) and q1 (MSB) of the 2-qubit subspace.
// The gate acts on the 4D subspace spanned by |q1 q0> in {00,01,10,11}.
void apply_2q(Statevector& sv, int n_qubits, const Gate2Q& G, int q0, int q1) {
    const int N = 1 << n_qubits;
    const int m0 = 1 << q0, m1 = 1 << q1;

    for (int i = 0; i < N; ++i) {
        // Only process each group of 4 once: skip unless both bits are 0
        if (i & m0 || i & m1) continue;
        const int i00 = i, i01 = i | m0, i10 = i | m1, i11 = i | m0 | m1;
        const cplx a00 = sv[i00], a01 = sv[i01],
                   a10 = sv[i10], a11 = sv[i11];
        sv[i00] = G[0]*a00 + G[1]*a01 + G[2]*a10  + G[3]*a11;
        sv[i01] = G[4]*a00 + G[5]*a01 + G[6]*a10  + G[7]*a11;
        sv[i10] = G[8]*a00 + G[9]*a01 + G[10]*a10 + G[11]*a11;
        sv[i11] = G[12]*a00+ G[13]*a01+ G[14]*a10 + G[15]*a11;
    }
}

void apply_cnot(Statevector& sv, int n_qubits, int control, int target) {
    const int N = 1 << n_qubits;
    const int ctrl_mask = 1 << control, tgt_mask = 1 << target;
    for (int i = 0; i < N; ++i) {
        if ((i & ctrl_mask) && !(i & tgt_mask))
            std::swap(sv[i], sv[i | tgt_mask]);
    }
}

void apply_cz(Statevector& sv, int n_qubits, int control, int target) {
    const int N = 1 << n_qubits;
    const int ctrl_mask = 1 << control, tgt_mask = 1 << target;
    for (int i = 0; i < N; ++i) {
        if ((i & ctrl_mask) && (i & tgt_mask))
            sv[i] = -sv[i];
    }
}

void apply_swap(Statevector& sv, int n_qubits, int q0, int q1) {
    const int N = 1 << n_qubits;
    const int m0 = 1 << q0, m1 = 1 << q1;
    for (int i = 0; i < N; ++i) {
        if ((i & m0) && !(i & m1))
            std::swap(sv[i], sv[(i & ~m0) | m1]);
    }
}

void apply_toffoli(Statevector& sv, int n_qubits, int c0, int c1, int target) {
    const int N = 1 << n_qubits;
    const int m0 = 1 << c0, m1 = 1 << c1, mt = 1 << target;
    for (int i = 0; i < N; ++i) {
        if ((i & m0) && (i & m1) && !(i & mt))
            std::swap(sv[i], sv[i | mt]);
    }
}

// Observables

real norm(const Statevector& sv) {
    return num::norm(sv);  // delegates to CVector overload
}

void normalize(Statevector& sv) {
    real n = num::norm(sv);
    for (auto& a : sv) a /= n;
}

std::vector<real> probabilities(const Statevector& sv) {
    std::vector<real> p(sv.size());
    for (std::size_t i = 0; i < sv.size(); ++i) p[i] = std::norm(sv[i]);
    return p;
}

real expectation(const Statevector& sv,
                 std::function<void(const Statevector&, Statevector&)> op) {
    Statevector out(sv.size(), cplx{0, 0});
    op(sv, out);
    cplx val{0, 0};
    for (std::size_t i = 0; i < sv.size(); ++i) val += std::conj(sv[i]) * out[i];
    return val.real();
}

real expectation_z(const Statevector& sv, int qubit) {
    const int mask = 1 << qubit;
    real val = 0;
    for (std::size_t i = 0; i < sv.size(); ++i) {
        real sign = (i & mask) ? real(-1) : real(1);
        val += sign * std::norm(sv[i]);
    }
    return val;
}

real expectation_x(const Statevector& sv, int n_qubits, int qubit) {
    // <X_q> = <sv | X_q | sv>  computed by applying X then taking inner product
    Statevector tmp = sv;
    apply_1q(tmp, n_qubits, gate_x(), qubit);
    cplx val{0, 0};
    for (std::size_t i = 0; i < sv.size(); ++i) val += std::conj(sv[i]) * tmp[i];
    return val.real();
}

real expectation_y(const Statevector& sv, int n_qubits, int qubit) {
    Statevector tmp = sv;
    apply_1q(tmp, n_qubits, gate_y(), qubit);
    cplx val{0, 0};
    for (std::size_t i = 0; i < sv.size(); ++i) val += std::conj(sv[i]) * tmp[i];
    return val.real();
}

std::array<cplx, 4> reduced_density_matrix(const Statevector& sv,
                                             int n_qubits, int qubit) {
    // Partial trace over all qubits except `qubit`.
    // rho_q[a,b] = Sigma_{env} alpha_{env|a} * conj(alpha_{env|b})
    // where env iterates over the N/2 basis states with qubit bit = 0.
    // This is O(N)  -- not O(N^2).
    const int N    = 1 << n_qubits;
    const int mask = 1 << qubit;
    std::array<cplx, 4> rho{};
    for (int i = 0; i < N; ++i) {
        if (i & mask) continue;           // visit each environment state once
        const int i1 = i | mask;          // same env, qubit = 1
        rho[0] += sv[i]  * std::conj(sv[i]);   // rho[0,0]
        rho[1] += sv[i]  * std::conj(sv[i1]);  // rho[0,1]
        rho[2] += sv[i1] * std::conj(sv[i]);   // rho[1,0]
        rho[3] += sv[i1] * std::conj(sv[i1]);  // rho[1,1]
    }
    return rho;
}

real entanglement_entropy(const Statevector& sv, int n_qubits, int qubit) {
    auto rho = reduced_density_matrix(sv, n_qubits, qubit);
    // Eigenvalues of 2x2 Hermitian: lambda = (tr +/- sqrt(tr^2-4det)) / 2
    real tr  = rho[0].real() + rho[3].real();
    real det = (rho[0] * rho[3] - rho[1] * rho[2]).real();
    real disc = std::max(real(0), tr * tr / 4 - det);
    real sq   = std::sqrt(disc);
    real l0 = tr / 2 + sq, l1 = tr / 2 - sq;
    auto xlogx = [](real x) -> real {
        return (x > 1e-15) ? -x * std::log2(x) : real(0);
    };
    return xlogx(l0) + xlogx(l1);
}

void apply_cy(Statevector& sv, int n_qubits, int control, int target) {
    // CY = controlled-Y: apply Y to target when control=1
    // Y = [[0,-i],[i,0]], so swap amplitudes with phase factor
    const int N = 1 << n_qubits;
    const int ctrl_mask = 1 << control, tgt_mask = 1 << target;
    for (int i = 0; i < N; ++i) {
        if ((i & ctrl_mask) && !(i & tgt_mask)) {
            int j = i | tgt_mask;
            cplx a0 = sv[i], a1 = sv[j];
            sv[i] = cplx{0, 1} * a1;   // i*a1
            sv[j] = cplx{0,-1} * a0;   // -i*a0
        }
    }
}

void apply_cp(Statevector& sv, int n_qubits, real lambda, int control, int target) {
    // Controlled phase: |11> -> e^{ilambda}|11>, others unchanged
    const int N = 1 << n_qubits;
    const int ctrl_mask = 1 << control, tgt_mask = 1 << target;
    const cplx phase{std::cos(lambda), std::sin(lambda)};
    for (int i = 0; i < N; ++i) {
        if ((i & ctrl_mask) && (i & tgt_mask))
            sv[i] *= phase;
    }
}

void apply_cswap(Statevector& sv, int n_qubits, int ctrl, int q0, int q1) {
    // Fredkin: swap q0 and q1 when ctrl=1
    const int N = 1 << n_qubits;
    const int mc = 1 << ctrl, m0 = 1 << q0, m1 = 1 << q1;
    for (int i = 0; i < N; ++i) {
        if ((i & mc) && (i & m0) && !(i & m1))
            std::swap(sv[i], sv[(i & ~m0) | m1]);
    }
}

} // namespace quantum
} // namespace num
