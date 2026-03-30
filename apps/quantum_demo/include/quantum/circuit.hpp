/// @file quantum/circuit.hpp
/// @brief Fluent quantum circuit builder and statevector simulator
///
/// Provides a chainable Circuit class that reads like a circuit diagram.
/// Gates are recorded and replayed on a statevector on demand, so you can
/// inspect intermediate states (step-through) or sample measurement outcomes.
///
/// ### Quick start
/// @code
/// #include "quantum/circuit.hpp"
/// using num::Circuit;
///
/// // Bell state
/// auto sv = Circuit(2).h(0).cx(0,1).statevector();
///
/// // Grover search (2 qubits, finds |11>)
/// auto result = Circuit(2)
///     .h(0).h(1)
///     .cz(0,1)                             // oracle
///     .h(0).h(1).x(0).x(1).cz(0,1).x(0).x(1).h(0).h(1)   // diffusion
///     .run(4096);
/// result.print();   // |11>: 4096/4096
///
/// // QFT on 3 qubits
/// Circuit qft(3);
/// qft.x(0)
///    .h(2).cp(M_PI/2, 1,2).cp(M_PI/4, 0,2)
///    .h(1).cp(M_PI/2, 0,1)
///    .h(0).swap(0,2);
/// qft.print();     // ASCII diagram
/// @endcode
#pragma once
#include "quantum/statevector.hpp"
#include <map>
#include <string>
#include <vector>

namespace num {

// Result

/// @brief Measurement outcome from Circuit::run()
struct Result {
    std::map<std::string, int> counts;  ///< basis label (e.g. "011") -> shot count
    quantum::Statevector sv;            ///< pre-measurement statevector
    int shots = 0;

    /// @brief Print counts sorted by count (descending)
    void print() const;

    /// @brief Basis label with the highest count
    std::string most_likely() const;

    /// @brief Empirical probability for a given label (e.g. "11")
    real probability(const std::string& label) const;
};

// GateView -- for renderers

/// @brief Compact gate description used by visualisation code
struct GateView {
    std::string label;  ///< e.g. "H", "CX", "RY(1.57)", "SWAP"
    int q0 = -1;        ///< primary qubit (target for 1Q; control for 2/3Q)
    int q1 = -1;        ///< secondary qubit (-1 if 1Q)
    int q2 = -1;        ///< tertiary qubit (-1 if not 3Q)
    int col = 0;        ///< time-slot column after layout pass
    /// 0 = single-qubit, 1 = controlled (q0=ctrl, q1=tgt),
    /// 2 = swap, 3 = Toffoli/Fredkin
    int kind = 0;
};

// Circuit

/// @brief Chainable quantum circuit builder
///
/// Gates are stored in order and applied to a fresh zero-state statevector
/// when execution is requested. The circuit itself is immutable between calls.
class Circuit {
public:
    explicit Circuit(int n_qubits);

    // Single-qubit gates

    Circuit& h  (int q);                              ///< Hadamard
    Circuit& x  (int q);                              ///< Pauli-X (NOT)
    Circuit& y  (int q);                              ///< Pauli-Y
    Circuit& z  (int q);                              ///< Pauli-Z
    Circuit& s  (int q);                              ///< Phase S = sqrtZ
    Circuit& sdg(int q);                              ///< Sdg
    Circuit& t  (int q);                              ///< T = sqrtS
    Circuit& tdg(int q);                              ///< Tdg
    Circuit& rx (real theta,  int q);                 ///< R_x(theta) = e^{-ithetaX/2}
    Circuit& ry (real theta,  int q);                 ///< R_y(theta) = e^{-ithetaY/2}
    Circuit& rz (real theta,  int q);                 ///< R_z(theta) = e^{-ithetaZ/2}
    Circuit& p  (real lambda, int q);                 ///< Phase P(lambda)
    Circuit& u  (real theta, real phi, real lambda, int q); ///< General SU(2)

    // Two-qubit gates

    Circuit& cx  (int ctrl, int tgt);                 ///< CNOT
    Circuit& cy  (int ctrl, int tgt);                 ///< Controlled-Y
    Circuit& cz  (int ctrl, int tgt);                 ///< Controlled-Z
    Circuit& cp  (real lambda, int ctrl, int tgt);    ///< Controlled phase
    Circuit& swap(int q0, int q1);                    ///< SWAP

    // Three-qubit gates

    Circuit& ccx  (int c0, int c1, int tgt);           ///< Toffoli (CCX)
    Circuit& cswap(int ctrl, int q0, int q1);          ///< Fredkin (CSWAP)

    // Custom gates

    Circuit& gate(const quantum::Gate&   G, int q,         std::string label = "U");
    Circuit& gate(const quantum::Gate2Q& G, int q0, int q1, std::string label = "U2");

    // Execution

    /// @brief Run full circuit -> statevector (no measurement collapse)
    quantum::Statevector statevector() const;

    /// @brief Apply first `n` gates only (0 = bare |0...0>).
    ///        Useful for stepping through the circuit in a visualiser.
    quantum::Statevector statevector_at(int n) const;

    /// @brief Sample `shots` measurements from the ideal distribution.
    ///        Uses pre-computed statevector probabilities  -- O(2^n + shots).
    Result run(int shots = 1024, unsigned seed = 42) const;

    // Display

    /// @brief ASCII wire diagram (Unicode box-drawing characters)
    std::string diagram() const;

    /// @brief Print diagram to stdout
    void print() const;

    // Inspection

    int n_qubits() const { return n_qubits_; }
    int n_gates()  const { return static_cast<int>(ops_.size()); }

    /// @brief Gate descriptions with layout columns  -- for renderers
    std::vector<GateView> views() const;

private:
    struct Op {
        enum Type {
            H, X, Y, Z, S, Sdg, T, Tdg,
            RX, RY, RZ, P, U,
            CX, CY, CZ, CP, SWAP,
            CCX, CSWAP,
            Custom1Q, Custom2Q
        } type;
        int  q0 = -1, q1 = -1, q2 = -1;
        real theta = 0, phi = 0, lambda = 0;
        quantum::Gate   g1{};
        quantum::Gate2Q g2{};
        std::string     label;   // only for Custom gates
    };

    int              n_qubits_;
    std::vector<Op>  ops_;

    void apply_op(const Op& op, quantum::Statevector& sv) const;

    // Returns (gate_col[i], total_cols)
    std::pair<std::vector<int>, int> compute_layout() const;
};

// Convenience factory functions

/// @brief Bell state |Phi+> = (|00> + |11>)/sqrt2 on qubits q0, q1
Circuit bell_pair(int q0 = 0, int q1 = 1);

/// @brief n-qubit GHZ state (|00...0> + |11...1>)/sqrt2
Circuit ghz_state(int n_qubits);

/// @brief n-qubit Quantum Fourier Transform circuit
///        Applied to |0...0> unless you prepend state-preparation gates.
Circuit qft_circuit(int n_qubits);

/// @brief Pretty-print a statevector, hiding amplitudes below threshold
void print_state(const quantum::Statevector& sv, int n_qubits,
                 real threshold = 1e-6);

} // namespace num
