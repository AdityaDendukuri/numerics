#include "quantum/circuit.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>

namespace num {

static constexpr real PI = 3.141592653589793238462643383279502884;

// Result

void Result::print() const {
    // Sort by count descending
    std::vector<std::pair<std::string,int>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });

    const int bar_width = 30;
    for (const auto& [label, cnt] : sorted) {
        if (cnt == 0) continue;
        double p = static_cast<double>(cnt) / shots;
        int    filled = static_cast<int>(p * bar_width + 0.5);
        std::string bar(filled, '#');
        bar += std::string(bar_width - filled, '.');
        std::printf("|%s>  [%s]  %5.1f%%  (%d)\n",
                    label.c_str(), bar.c_str(), p * 100.0, cnt);
    }
}

std::string Result::most_likely() const {
    std::string best;
    int best_cnt = -1;
    for (const auto& [label, cnt] : counts)
        if (cnt > best_cnt) { best_cnt = cnt; best = label; }
    return best;
}

real Result::probability(const std::string& label) const {
    auto it = counts.find(label);
    if (it == counts.end()) return 0;
    return static_cast<real>(it->second) / shots;
}

// Circuit constructor

Circuit::Circuit(int n_qubits) : n_qubits_(n_qubits) {}

// Single-qubit gate methods

Circuit& Circuit::h  (int q) { ops_.push_back({Op::H,   q}); return *this; }
Circuit& Circuit::x  (int q) { ops_.push_back({Op::X,   q}); return *this; }
Circuit& Circuit::y  (int q) { ops_.push_back({Op::Y,   q}); return *this; }
Circuit& Circuit::z  (int q) { ops_.push_back({Op::Z,   q}); return *this; }
Circuit& Circuit::s  (int q) { ops_.push_back({Op::S,   q}); return *this; }
Circuit& Circuit::sdg(int q) { ops_.push_back({Op::Sdg, q}); return *this; }
Circuit& Circuit::t  (int q) { ops_.push_back({Op::T,   q}); return *this; }
Circuit& Circuit::tdg(int q) { ops_.push_back({Op::Tdg, q}); return *this; }

Circuit& Circuit::rx(real theta, int q) {
    Op op{Op::RX, q}; op.theta = theta; ops_.push_back(op); return *this;
}
Circuit& Circuit::ry(real theta, int q) {
    Op op{Op::RY, q}; op.theta = theta; ops_.push_back(op); return *this;
}
Circuit& Circuit::rz(real theta, int q) {
    Op op{Op::RZ, q}; op.theta = theta; ops_.push_back(op); return *this;
}
Circuit& Circuit::p(real lambda, int q) {
    Op op{Op::P, q}; op.lambda = lambda; ops_.push_back(op); return *this;
}
Circuit& Circuit::u(real theta, real phi, real lambda, int q) {
    Op op{Op::U, q}; op.theta = theta; op.phi = phi; op.lambda = lambda;
    ops_.push_back(op); return *this;
}

// Two-qubit gate methods

Circuit& Circuit::cx  (int ctrl, int tgt) {
    Op op{Op::CX, ctrl, tgt}; ops_.push_back(op); return *this;
}
Circuit& Circuit::cy  (int ctrl, int tgt) {
    Op op{Op::CY, ctrl, tgt}; ops_.push_back(op); return *this;
}
Circuit& Circuit::cz  (int ctrl, int tgt) {
    Op op{Op::CZ, ctrl, tgt}; ops_.push_back(op); return *this;
}
Circuit& Circuit::cp  (real lambda, int ctrl, int tgt) {
    Op op{Op::CP, ctrl, tgt}; op.lambda = lambda;
    ops_.push_back(op); return *this;
}
Circuit& Circuit::swap(int q0, int q1) {
    Op op{Op::SWAP, q0, q1}; ops_.push_back(op); return *this;
}

// Three-qubit gate methods

Circuit& Circuit::ccx(int c0, int c1, int tgt) {
    Op op{Op::CCX, c0, c1, tgt}; ops_.push_back(op); return *this;
}
Circuit& Circuit::cswap(int ctrl, int q0, int q1) {
    Op op{Op::CSWAP, ctrl, q0, q1}; ops_.push_back(op); return *this;
}

// Custom gates

Circuit& Circuit::gate(const quantum::Gate& G, int q, std::string label) {
    Op op{Op::Custom1Q, q};
    op.g1 = G; op.label = std::move(label);
    ops_.push_back(op); return *this;
}
Circuit& Circuit::gate(const quantum::Gate2Q& G, int q0, int q1, std::string label) {
    Op op{Op::Custom2Q, q0, q1};
    op.g2 = G; op.label = std::move(label);
    ops_.push_back(op); return *this;
}

// Gate dispatch

void Circuit::apply_op(const Op& op, quantum::Statevector& sv) const {
    using namespace quantum;
    switch (op.type) {
    case Op::H:       apply_1q(sv, n_qubits_, gate_h(),               op.q0); break;
    case Op::X:       apply_1q(sv, n_qubits_, gate_x(),               op.q0); break;
    case Op::Y:       apply_1q(sv, n_qubits_, gate_y(),               op.q0); break;
    case Op::Z:       apply_1q(sv, n_qubits_, gate_z(),               op.q0); break;
    case Op::S:       apply_1q(sv, n_qubits_, gate_s(),               op.q0); break;
    case Op::Sdg:     apply_1q(sv, n_qubits_, gate_sdg(),             op.q0); break;
    case Op::T:       apply_1q(sv, n_qubits_, gate_t(),               op.q0); break;
    case Op::Tdg:     apply_1q(sv, n_qubits_, gate_tdg(),             op.q0); break;
    case Op::RX:      apply_1q(sv, n_qubits_, gate_rx(op.theta),      op.q0); break;
    case Op::RY:      apply_1q(sv, n_qubits_, gate_ry(op.theta),      op.q0); break;
    case Op::RZ:      apply_1q(sv, n_qubits_, gate_rz(op.theta),      op.q0); break;
    case Op::P:       apply_1q(sv, n_qubits_, gate_p(op.lambda),      op.q0); break;
    case Op::U:       apply_1q(sv, n_qubits_,
                                gate_u(op.theta, op.phi, op.lambda),  op.q0); break;

    case Op::CX:      apply_cnot (sv, n_qubits_, op.q0, op.q1); break;
    case Op::CY:      apply_cy   (sv, n_qubits_, op.q0, op.q1); break;
    case Op::CZ:      apply_cz   (sv, n_qubits_, op.q0, op.q1); break;
    case Op::CP:      apply_cp   (sv, n_qubits_, op.lambda, op.q0, op.q1); break;
    case Op::SWAP:    apply_swap (sv, n_qubits_, op.q0, op.q1); break;

    case Op::CCX:     apply_toffoli(sv, n_qubits_, op.q0, op.q1, op.q2); break;
    case Op::CSWAP:   apply_cswap  (sv, n_qubits_, op.q0, op.q1, op.q2); break;

    case Op::Custom1Q: apply_1q(sv, n_qubits_, op.g1, op.q0); break;
    case Op::Custom2Q: apply_2q(sv, n_qubits_, op.g2, op.q0, op.q1); break;
    }
}

// Execution

quantum::Statevector Circuit::statevector() const {
    return statevector_at(static_cast<int>(ops_.size()));
}

quantum::Statevector Circuit::statevector_at(int n) const {
    auto sv = quantum::zero_state(n_qubits_);
    int limit = std::min(n, static_cast<int>(ops_.size()));
    for (int i = 0; i < limit; ++i)
        apply_op(ops_[i], sv);
    return sv;
}

Result Circuit::run(int shots, unsigned seed) const {
    auto sv    = statevector();
    auto probs = quantum::probabilities(sv);

    std::mt19937 rng(seed);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    // Build zero-count map for all states so Result::print shows full histogram
    Result result;
    result.sv    = sv;
    result.shots = shots;
    for (int k = 0; k < (1 << n_qubits_); ++k) {
        std::string label(n_qubits_, '0');
        for (int b = 0; b < n_qubits_; ++b)
            label[n_qubits_ - 1 - b] = ((k >> b) & 1) ? '1' : '0';
        result.counts[label] = 0;
    }
    for (int i = 0; i < shots; ++i) {
        int k = dist(rng);
        std::string label(n_qubits_, '0');
        for (int b = 0; b < n_qubits_; ++b)
            label[n_qubits_ - 1 - b] = ((k >> b) & 1) ? '1' : '0';
        result.counts[label]++;
    }
    return result;
}

// Layout pass

std::pair<std::vector<int>, int> Circuit::compute_layout() const {
    std::vector<int> last(n_qubits_, -1);
    std::vector<int> gate_col(ops_.size(), 0);
    int total = 0;

    for (int i = 0; i < (int)ops_.size(); ++i) {
        const auto& op = ops_[i];
        int c = last[op.q0];
        if (op.q1 >= 0) c = std::max(c, last[op.q1]);
        if (op.q2 >= 0) c = std::max(c, last[op.q2]);
        gate_col[i] = c + 1;
        total = std::max(total, gate_col[i] + 1);
        last[op.q0] = gate_col[i];
        if (op.q1 >= 0) last[op.q1] = gate_col[i];
        if (op.q2 >= 0) last[op.q2] = gate_col[i];
    }
    return {gate_col, total};
}

// views()

std::vector<GateView> Circuit::views() const {
    auto [gate_col, total_cols] = compute_layout();

    auto fmt_angle = [](real theta) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", theta);
        return std::string(buf);
    };

    std::vector<GateView> out;
    out.reserve(ops_.size());

    for (int i = 0; i < (int)ops_.size(); ++i) {
        const auto& op = ops_[i];
        GateView v;
        v.q0  = op.q0; v.q1 = op.q1; v.q2 = op.q2;
        v.col = gate_col[i];

        switch (op.type) {
        case Op::H:    v.label = "H";  v.kind = 0; break;
        case Op::X:    v.label = "X";  v.kind = 0; break;
        case Op::Y:    v.label = "Y";  v.kind = 0; break;
        case Op::Z:    v.label = "Z";  v.kind = 0; break;
        case Op::S:    v.label = "S";  v.kind = 0; break;
        case Op::Sdg:  v.label = "Sdg"; v.kind = 0; break;
        case Op::T:    v.label = "T";  v.kind = 0; break;
        case Op::Tdg:  v.label = "Tdg"; v.kind = 0; break;
        case Op::RX:   v.label = "Rx(" + fmt_angle(op.theta) + ")";  v.kind = 0; break;
        case Op::RY:   v.label = "Ry(" + fmt_angle(op.theta) + ")";  v.kind = 0; break;
        case Op::RZ:   v.label = "Rz(" + fmt_angle(op.theta) + ")";  v.kind = 0; break;
        case Op::P:    v.label = "P(" + fmt_angle(op.lambda) + ")";  v.kind = 0; break;
        case Op::U:    v.label = "U";  v.kind = 0; break;
        case Op::CX:   v.label = "CX";    v.kind = 1; break;
        case Op::CY:   v.label = "CY";    v.kind = 1; break;
        case Op::CZ:   v.label = "CZ";    v.kind = 1; break;
        case Op::CP:   v.label = "CP(" + fmt_angle(op.lambda) + ")"; v.kind = 1; break;
        case Op::SWAP: v.label = "SWAP";  v.kind = 2; break;
        case Op::CCX:  v.label = "CCX";   v.kind = 3; break;
        case Op::CSWAP:v.label = "CSWAP"; v.kind = 3; break;
        case Op::Custom1Q: v.label = op.label; v.kind = 0; break;
        case Op::Custom2Q: v.label = op.label; v.kind = 1; break;
        }
        out.push_back(v);
    }
    return out;
}

// ASCII diagram

std::string Circuit::diagram() const {
    if (ops_.empty()) {
        std::string out;
        for (int q = 0; q < n_qubits_; ++q)
            out += "q[" + std::to_string(q) + "]: ---\n";
        return out;
    }

    auto [gate_col, total_cols] = compute_layout();

    // Grid: 2*n_qubits-1 display rows (qubit rows interleaved with connector rows)
    // Each cell is a fixed 3-char string
    const int disp_rows = 2 * n_qubits_ - 1;
    std::vector<std::vector<std::string>> grid(
        disp_rows, std::vector<std::string>(total_cols, "---"));
    // Connector rows default to empty space
    for (int r = 1; r < disp_rows; r += 2)
        for (int c = 0; c < total_cols; ++c)
            grid[r][c] = "   ";

    // Helper: add vertical line through intermediate qubits
    auto add_vertical = [&](int q_lo, int q_hi, int col) {
        for (int q = q_lo; q < q_hi; ++q)
            grid[2*q + 1][col] = " | ";         // connector row
        for (int q = q_lo + 1; q < q_hi; ++q)
            grid[2*q][col] = "-+-";             // crossing wire
    };

    for (int i = 0; i < (int)ops_.size(); ++i) {
        const auto& op = ops_[i];
        int c = gate_col[i];

        auto set = [&](int q, std::string sym) { grid[2*q][c] = sym; };

        switch (op.type) {
        case Op::H:    set(op.q0, "-H-"); break;
        case Op::X:    set(op.q0, "-X-"); break;
        case Op::Y:    set(op.q0, "-Y-"); break;
        case Op::Z:    set(op.q0, "-Z-"); break;
        case Op::S:    set(op.q0, "-S-"); break;
        case Op::Sdg:  set(op.q0, "Sdg-"); break;
        case Op::T:    set(op.q0, "-T-"); break;
        case Op::Tdg:  set(op.q0, "Tdg-"); break;
        case Op::RX:   set(op.q0, "Rx-"); break;
        case Op::RY:   set(op.q0, "Ry-"); break;
        case Op::RZ:   set(op.q0, "Rz-"); break;
        case Op::P:    set(op.q0, "-P-"); break;
        case Op::U:    set(op.q0, "-U-"); break;
        case Op::CX:
            set(op.q0, "-*-");
            set(op.q1, "-(+)-");
            add_vertical(std::min(op.q0,op.q1), std::max(op.q0,op.q1), c);
            break;
        case Op::CY:
            set(op.q0, "-*-");
            set(op.q1, "-Y-");
            add_vertical(std::min(op.q0,op.q1), std::max(op.q0,op.q1), c);
            break;
        case Op::CZ:
            set(op.q0, "-*-");
            set(op.q1, "-*-");
            add_vertical(std::min(op.q0,op.q1), std::max(op.q0,op.q1), c);
            break;
        case Op::CP:
            set(op.q0, "-*-");
            set(op.q1, "-P-");
            add_vertical(std::min(op.q0,op.q1), std::max(op.q0,op.q1), c);
            break;
        case Op::SWAP: {
            set(op.q0, "-x-");
            set(op.q1, "-x-");
            add_vertical(std::min(op.q0,op.q1), std::max(op.q0,op.q1), c);
            break;
        }
        case Op::CCX:
            set(op.q0, "-*-");
            set(op.q1, "-*-");
            set(op.q2, "-(+)-");
            add_vertical(std::min({op.q0,op.q1,op.q2}),
                         std::max({op.q0,op.q1,op.q2}), c);
            break;
        case Op::CSWAP:
            set(op.q0, "-*-");
            set(op.q1, "-x-");
            set(op.q2, "-x-");
            add_vertical(std::min({op.q0,op.q1,op.q2}),
                         std::max({op.q0,op.q1,op.q2}), c);
            break;
        case Op::Custom1Q: {
            std::string sym = op.label.substr(0, 1);
            set(op.q0, "-" + sym + "-");
            break;
        }
        case Op::Custom2Q:
            set(op.q0, "-*-");
            set(op.q1, "-" + op.label.substr(0,1) + "-");
            add_vertical(std::min(op.q0,op.q1), std::max(op.q0,op.q1), c);
            break;
        }
    }

    std::string out;
    for (int q = 0; q < n_qubits_; ++q) {
        std::string prefix = "q[" + std::to_string(q) + "]: ";
        out += prefix;
        for (int c = 0; c < total_cols; ++c)
            out += grid[2*q][c];
        out += "-\n";

        if (q < n_qubits_ - 1) {
            out += std::string(prefix.size(), ' ');
            for (int c = 0; c < total_cols; ++c)
                out += grid[2*q + 1][c];
            out += "\n";
        }
    }
    return out;
}

void Circuit::print() const { std::cout << diagram(); }

// Factory functions

Circuit bell_pair(int q0, int q1) {
    int n = std::max(q0, q1) + 1;
    Circuit c(n);
    return c.h(q0).cx(q0, q1);
}

Circuit ghz_state(int n_qubits) {
    Circuit c(n_qubits);
    c.h(0);
    for (int q = 1; q < n_qubits; ++q)
        c.cx(0, q);
    return c;
}

Circuit qft_circuit(int n_qubits) {
    Circuit c(n_qubits);
    for (int k = n_qubits - 1; k >= 0; --k) {
        c.h(k);
        for (int j = k - 1; j >= 0; --j)
            c.cp(PI / (1 << (k - j)), j, k);
    }
    // Bit-reversal swap network
    for (int i = 0; i < n_qubits / 2; ++i)
        c.swap(i, n_qubits - 1 - i);
    return c;
}

void print_state(const quantum::Statevector& sv, int n_qubits, real threshold) {
    int N = 1 << n_qubits;
    for (int k = 0; k < N; ++k) {
        if (std::abs(sv[k]) < threshold) continue;
        // Build label (MSB=q[n-1], LSB=q[0])
        std::string label(n_qubits, '0');
        for (int b = 0; b < n_qubits; ++b)
            label[n_qubits - 1 - b] = ((k >> b) & 1) ? '1' : '0';
        double p = std::norm(sv[k]);
        std::printf("|%s>  %+.4f%+.4fi  (%.1f%%)\n",
                    label.c_str(), sv[k].real(), sv[k].imag(), p * 100.0);
    }
}

} // namespace num
