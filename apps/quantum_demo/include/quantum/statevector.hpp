/// @file quantum/statevector.hpp
/// @brief Statevector-based quantum circuit simulator
///
/// Simulates an n-qubit system as a dense complex statevector of length 2^n.
/// Qubit indexing is little-endian: qubit 0 is the least-significant bit of
/// the basis-state index.
///
/// @code
/// using namespace num::quantum;
/// auto sv = zero_state(2);          // |00>
/// apply_1q(sv, 2, gate_h(), 0);     // H on qubit 0  ->  (|00> + |10>) / sqrt2
/// apply_cnot(sv, 2, 0, 1);          // CNOT(ctrl=0, tgt=1) -> Bell state
/// auto p = probabilities(sv);       // {0.5, 0, 0, 0.5}
/// @endcode
#pragma once
#include "core/types.hpp"
#include "core/vector.hpp"
#include <array>
#include <functional>
#include <vector>

namespace num {
namespace quantum {

// Types

/// @brief Dense statevector: complex vector of length 2^n_qubits, backed by
/// num::CVector
using Statevector = num::CVector;

/// @brief 2x2 single-qubit gate stored row-major: { G[0,0], G[0,1], G[1,0],
/// G[1,1] }
using Gate = std::array<cplx, 4>;

/// @brief 4x4 two-qubit gate stored row-major (16 entries)
using Gate2Q = std::array<cplx, 16>;

// State construction

/// @brief Returns |0...0> computational basis state for n_qubits qubits
Statevector zero_state(int n_qubits);

/// @brief Returns the computational basis state |k> for an n_qubits system
Statevector basis_state(int n_qubits, int k);

// Single-qubit gates

Gate gate_x();            ///< Pauli-X  (NOT)
Gate gate_y();            ///< Pauli-Y
Gate gate_z();            ///< Pauli-Z
Gate gate_h();            ///< Hadamard
Gate gate_s();            ///< Phase gate S = diag(1, i)
Gate gate_sdg();          ///< Sdg = diag(1, -i)
Gate gate_t();            ///< T gate = diag(1, e^{ipi/4})
Gate gate_tdg();          ///< Tdg gate
Gate gate_rx(real theta); ///< Rotation about X: R_x(theta) = exp(-ithetaX/2)
Gate gate_ry(real theta); ///< Rotation about Y: R_y(theta) = exp(-ithetaY/2)
Gate gate_rz(real theta); ///< Rotation about Z: R_z(theta) = exp(-ithetaZ/2)
Gate gate_p(real lambda); ///< Phase gate: diag(1, e^{ilambda})
Gate gate_u(real theta, real phi, real lambda); ///< General SU(2) gate

// Two-qubit gates

Gate2Q
gate2q_cnot();      ///< CNOT (qubit 0 = control, qubit 1 = target in subspace)
Gate2Q gate2q_cz(); ///< Controlled-Z
Gate2Q gate2q_swap(); ///< SWAP

// Gate application

/// @brief Apply a 2x2 gate to a single qubit in-place
/// @param sv        Statevector of length 2^n_qubits (modified in-place)
/// @param n_qubits  Total number of qubits
/// @param G         2x2 gate (row-major)
/// @param target    Target qubit index (0 = LSB)
void apply_1q(Statevector& sv, int n_qubits, const Gate& G, int target);

/// @brief Apply a general 4x4 two-qubit gate to qubits q0, q1
/// @param sv        Statevector
/// @param n_qubits  Total number of qubits
/// @param G         4x4 gate acting on (q0, q1) in little-endian order
/// @param q0        First qubit (LSB of the 2-qubit subspace)
/// @param q1        Second qubit (MSB of the 2-qubit subspace)
void apply_2q(Statevector& sv, int n_qubits, const Gate2Q& G, int q0, int q1);

/// @brief CNOT with explicit control and target qubits
void apply_cnot(Statevector& sv, int n_qubits, int control, int target);

/// @brief Controlled-Z
void apply_cz(Statevector& sv, int n_qubits, int control, int target);

/// @brief SWAP two qubits
void apply_swap(Statevector& sv, int n_qubits, int q0, int q1);

/// @brief Toffoli (CCX) gate
void apply_toffoli(Statevector& sv, int n_qubits, int c0, int c1, int target);

/// @brief Controlled-Y
void apply_cy(Statevector& sv, int n_qubits, int control, int target);

/// @brief Controlled phase: applies phase e^{ilambda} when both control=1 and
/// target=1
void apply_cp(Statevector& sv,
              int          n_qubits,
              real         lambda,
              int          control,
              int          target);

/// @brief Fredkin (CSWAP): swaps q0 and q1 when ctrl=1
void apply_cswap(Statevector& sv, int n_qubits, int ctrl, int q0, int q1);

// Observables

/// @brief L2 norm of the statevector (should be 1 for a valid quantum state)
real norm(const Statevector& sv);

/// @brief Normalize statevector in-place
void normalize(Statevector& sv);

/// @brief Measurement probability for each basis state: p_i = |a_i|^2
std::vector<real> probabilities(const Statevector& sv);

/// @brief Expectation value <psi|O|psi> for a Hermitian operator given as
/// matrix-free matvec
real expectation(const Statevector&                                    sv,
                 std::function<void(const Statevector&, Statevector&)> op);

/// @brief <Z_q> = P(q=0) - P(q=1) for qubit q
real expectation_z(const Statevector& sv, int qubit);

/// @brief <X_q> for qubit q
real expectation_x(const Statevector& sv, int n_qubits, int qubit);

/// @brief <Y_q> for qubit q
real expectation_y(const Statevector& sv, int n_qubits, int qubit);

/// @brief Reduced density matrix for a single qubit (2x2, returned row-major)
std::array<std::complex<real>, 4> reduced_density_matrix(const Statevector& sv,
                                                         int n_qubits,
                                                         int qubit);

/// @brief Von Neumann entropy of a single qubit's reduced density matrix
real entanglement_entropy(const Statevector& sv, int n_qubits, int qubit);

} // namespace quantum
} // namespace num
