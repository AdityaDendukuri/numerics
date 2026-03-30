<!--! @page page_app_quantum Quantum Circuit Simulator -->

# Quantum Circuit Simulator

Interactive statevector-based quantum circuit simulator with step-by-step gate animation, live probability histogram, and entanglement tracking. Demonstrates five canonical circuits — Bell state, GHZ state, Grover search, quantum teleportation, and the Quantum Fourier Transform — using the `num::Circuit` fluent API built on `num::quantum`.

---

## Physical Model

An $n$-qubit quantum system lives in a $2^n$-dimensional complex Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$. Its state is a **statevector**

$$|\psi\rangle = \sum_{k=0}^{2^n - 1} \alpha_k |k\rangle, \qquad \alpha_k \in \mathbb{C}, \quad \sum_k |\alpha_k|^2 = 1$$

where $|k\rangle$ are the computational basis states (bit-strings of length $n$). This implementation uses **little-endian** qubit ordering: qubit 0 is the least-significant bit of the basis index, so $|k\rangle = |q_{n-1} \cdots q_1 q_0\rangle$ with $k = \sum_i q_i 2^i$.

**Storage**: the statevector is a contiguous `std::vector<std::complex<double>>` of length $2^n$, requiring $16 \cdot 2^n$ bytes ($\approx 512\,\text{MB}$ at $n = 25$). Practical simulation is tractable to about $n = 28$ on a workstation.

---

## Quantum Circuit Model

A quantum circuit is a sequence of **unitary gates** applied to the initial state $|0\cdots0\rangle$:

$$|\psi_\text{final}\rangle = U_d \cdots U_2 U_1 |0\cdots0\rangle$$

Each gate $U_i$ acts on one, two, or three qubits; on the remaining qubits it acts as the identity $I$. A single-qubit gate $G \in U(2)$ acting on qubit $t$ in an $n$-qubit system is realised by partitioning the $2^n$ amplitudes into $2^{n-1}$ pairs

$$\bigl(\alpha_{k},\, \alpha_{k + 2^t}\bigr), \quad k \;\text{with bit}\; t = 0,$$

and applying $G$ to each pair independently. This costs $O(2^n)$ time and $O(1)$ extra space per gate.

**Measurement** collapses $|\psi\rangle$ to basis state $|k\rangle$ with probability $|\alpha_k|^2$. In the ideal simulation model all shots are sampled from the final distribution in $O(\text{shots})$ time without re-running the circuit.

---

## Gate Set

### Single-qubit gates

| Gate | Matrix | Description |
|------|--------|-------------|
| $H$ | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | Hadamard — creates equal superposition |
| $X$ | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | Pauli-X (bit flip) |
| $Y$ | $\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ | Pauli-Y |
| $Z$ | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ | Pauli-Z (phase flip) |
| $S$ | $\begin{pmatrix}1&0\\0&i\end{pmatrix}$ | Phase $S = \sqrt{Z}$ |
| $T$ | $\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}$ | $\frac{\pi}{8}$ gate, $T = \sqrt{S}$ |
| $R_x(\theta)$ | $\begin{pmatrix}\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}$ | Rotation about $X$ |
| $R_y(\theta)$ | $\begin{pmatrix}\cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}$ | Rotation about $Y$ |
| $R_z(\theta)$ | $\begin{pmatrix}e^{-i\theta/2}&0\\0&e^{i\theta/2}\end{pmatrix}$ | Rotation about $Z$ |
| $P(\lambda)$ | $\begin{pmatrix}1&0\\0&e^{i\lambda}\end{pmatrix}$ | Arbitrary phase |
| $U(\theta,\phi,\lambda)$ | $\begin{pmatrix}\cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2}\end{pmatrix}$ | General SU(2) |

### Two-qubit gates

| Gate | Action | Description |
|------|--------|-------------|
| CX (CNOT) | $|c, t\rangle \to |c,\, t \oplus c\rangle$ | Flip target when control = 1 |
| CY | $|c, t\rangle \to |c,\, Y_t\, c\rangle$ | Controlled-$Y$ |
| CZ | $|c, t\rangle \to (-1)^{c \cdot t}\,|c, t\rangle$ | Phase flip when both qubits = 1 |
| CP($\lambda$) | $|11\rangle \to e^{i\lambda}|11\rangle$ | Controlled phase |
| SWAP | $|a,b\rangle \to |b,a\rangle$ | Exchange two qubits |

### Three-qubit gates

| Gate | Action |
|------|--------|
| CCX (Toffoli) | Flip target when both controls = 1 |
| CSWAP (Fredkin) | Swap two qubits when control = 1 |

---

## Preset Circuits

### 1 — Bell State $|\Phi^+\rangle$  (2 qubits)

The simplest maximally entangled state:

$$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$

**Circuit:**

$$|00\rangle \xrightarrow{H_0} \frac{|00\rangle+|10\rangle}{\sqrt{2}} \xrightarrow{\text{CX}_{0\to1}} \frac{|00\rangle+|11\rangle}{\sqrt{2}}$$

After both gates the entanglement entropy $S(q_0) = 1$ ebit (maximum for a qubit), and measurements always yield $|00\rangle$ or $|11\rangle$ with equal probability.

---

### 2 — GHZ State  (3 qubits)

The Greenberger–Horne–Zeilinger state generalises Bell entanglement to $n$ qubits:

$$|GHZ\rangle = \frac{|000\rangle + |111\rangle}{\sqrt{2}}$$

**Circuit:**

$$|000\rangle \xrightarrow{H_0} \xrightarrow{\text{CX}_{0\to1}} \xrightarrow{\text{CX}_{0\to2}} \frac{|000\rangle+|111\rangle}{\sqrt{2}}$$

All three qubits are pairwise maximally entangled. Measuring any one qubit instantly determines the other two — a signature of non-local quantum correlations. With standard local hidden-variable models this is impossible to reproduce.

---

### 3 — Grover Search  (2 qubits, target $|11\rangle$)

Grover's algorithm finds a marked element in an unstructured database of $N = 2^n$ items in $O(\sqrt{N})$ queries, versus $O(N)$ classically. For $n = 2$, a single iteration achieves 100% success probability.

**Oracle** $U_\omega$: applies a phase flip to the target state $|\omega\rangle = |11\rangle$. Implemented as CZ$(q_0, q_1)$:

$$U_\omega = I - 2|\omega\rangle\langle\omega|, \qquad U_\omega|11\rangle = -|11\rangle$$

**Diffusion operator** $D = 2|s\rangle\langle s| - I$ where $|s\rangle = H^{\otimes 2}|00\rangle$:

$$D = H^{\otimes 2} \bigl(2|00\rangle\langle00| - I\bigr) H^{\otimes 2}$$

The inner reflection $2|00\rangle\langle 00| - I$ is implemented as $X^{\otimes 2} \cdot \text{CZ} \cdot X^{\otimes 2}$.

**Full circuit** (10 gates):

$$|00\rangle \xrightarrow{H\otimes H} \xrightarrow{U_\omega} \xrightarrow{D} |11\rangle$$

**Amplitude amplification** — after initialisation the target amplitude is $\frac{1}{2}$. The oracle rotates the state by $2\arcsin\!\bigl(\frac{1}{2}\bigr) = \frac{\pi}{3}$ in the two-dimensional $\{|s\rangle, |\omega\rangle\}$ subspace, and the diffusion operator reflects it onto $|\omega\rangle$ exactly.

---

### 4 — Quantum Teleportation  (3 qubits)

Transmits an unknown qubit state from Alice ($q_0$) to Bob ($q_2$) using a pre-shared Bell pair and two classical bits. The state prepared on $q_0$ is $|\phi\rangle = R_y(\pi/3)|0\rangle = \cos(\pi/6)|0\rangle + \sin(\pi/6)|1\rangle$.

**Protocol:**

| Step | Gates | Effect |
|------|-------|--------|
| Prepare | $R_y(\pi/3)$ on $q_0$ | $|\phi\rangle \otimes |00\rangle$ |
| Bell pair | $H_1,\ \text{CX}_{1\to2}$ | $|\phi\rangle \otimes |\Phi^+\rangle_{12}$ |
| Alice encodes | $\text{CX}_{0\to1},\ H_0$ | Entangles $q_0$ with Bell pair |
| Bob corrects | $\text{CX}_{1\to2},\ \text{CZ}_{0\to2}$ | $q_2$ collapses to $|\phi\rangle$ |

After all seven gates the reduced state of $q_2$ is $|\phi\rangle$ regardless of the unmeasured values of $q_0$ and $q_1$. No quantum information travelled faster than light — the classical correction bits carry the required information.

The fidelity of the teleportation is verified by the probability distribution on $q_2$: $P(q_2 = 0) = \cos^2(\pi/6) \approx 75\%$, $P(q_2 = 1) = \sin^2(\pi/6) = 25\%$.

---

### 5 — Quantum Fourier Transform  (3 qubits, input $|001\rangle$)

The QFT maps computational basis states to frequency basis states:

$$\text{QFT}_N |j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i jk/N} |k\rangle, \qquad N = 2^n$$

For $n = 3$, $N = 8$. The input is $|001\rangle$ ($j = 1$), prepared with an $X$ gate on $q_0$.

**Circuit** (8 gates):

$$X_0 \to H_2 \to \text{CP}(\pi/2)_{1\to2} \to \text{CP}(\pi/4)_{0\to2} \to H_1 \to \text{CP}(\pi/2)_{0\to1} \to H_0 \to \text{SWAP}(0,2)$$

The controlled-phase gate $\text{CP}(\lambda)$ contributes the $e^{2\pi i jk/N}$ phase factors. The final SWAP implements the bit-reversal permutation required for the little-endian output convention.

**Output** for input $|j=1\rangle$: all eight basis states have equal amplitude $\frac{1}{\sqrt{8}}$ and probability $\frac{1}{8} = 12.5\%$, but with phases $e^{2\pi i k/8}$ encoding the frequency $j=1$ — the QFT signature of a unit-frequency signal.

**Circuit depth**: $O(n^2)$ for $n$ qubits (this circuit), versus $O(n \log n)$ for the best classical FFT. The quantum advantage is in the exponentially larger state space processed coherently.

---

## Simulation Model

### Statevector evolution

Each gate is applied in-place to the statevector in $O(2^n)$ time. The circuit stores gates as a typed list; `statevector_at(k)` applies the first $k$ gates to a fresh $|0\cdots0\rangle$ state, enabling step-through without re-storing intermediate vectors.

### Measurement sampling

`Circuit::run(shots)` computes the statevector **once**, extracts probabilities $p_k = |\alpha_k|^2$, then draws `shots` samples from $\text{Categorical}(p_0, \ldots, p_{2^n-1})$ using `std::discrete_distribution`. Total cost is $O(\text{depth} \cdot 2^n + \text{shots})$.

### Observables

**Entanglement entropy** of qubit $q$ is the von Neumann entropy of its reduced density matrix $\rho_q = \text{Tr}_{\bar{q}}|\psi\rangle\langle\psi|$:

$$S(q) = -\text{Tr}(\rho_q \log_2 \rho_q) = -\lambda_0 \log_2 \lambda_0 - \lambda_1 \log_2 \lambda_1$$

where $\lambda_0, \lambda_1$ are the eigenvalues of the $2\times 2$ matrix $\rho_q$. $S = 0$ means $q$ is unentangled; $S = 1$ ebit is maximal entanglement.

**Expectation values**: $\langle Z_q \rangle = P(q=0) - P(q=1)$ computed directly from probabilities. $\langle X_q \rangle$ and $\langle Y_q \rangle$ are computed by applying the gate to a copy of the statevector and taking the inner product.

---

## Numerics Library Integration

| Feature | Where used |
|---------|-----------|
| `num::quantum::Statevector` | Dense $2^n$ complex amplitude vector |
| `num::quantum::apply_1q` | All single-qubit gates in $O(2^n)$ |
| `num::quantum::apply_cnot/cz/swap` | Optimised 2-qubit gates (no 4×4 matrix multiply) |
| `num::quantum::apply_cp` | Controlled phase for QFT and teleportation |
| `num::quantum::probabilities` | $|\alpha_k|^2$ histogram for sampling |
| `num::quantum::reduced_density_matrix` | $2\times2$ partial trace over $n-1$ qubits |
| `num::quantum::entanglement_entropy` | Von Neumann entropy via $2\times2$ eigenvalues |
| `num::Circuit` | Fluent gate builder + layout pass + diagram renderer |
| `num::GateView` | Gate metadata consumed by the raylib renderer |

---

## The `num::Circuit` API

The `Circuit` class is the primary library interface. All gate methods return `*this` for chaining:

```cpp
#include "operator/circuit.hpp"
using namespace num;

// Bell state — one expression
auto sv = Circuit(2).h(0).cx(0, 1).statevector();

// Sample 4096 shots
auto result = Circuit(2).h(0).cx(0, 1).run(4096);
result.print();
// |00⟩  [###############.]  50.1%  (2049)
// |11⟩  [##############..]  49.9%  (2047)

// ASCII circuit diagram
Circuit(3).h(0).cx(0,1).cx(0,2).print();
// q[0]: ─H─●─●─
//          │ │
// q[1]: ───⊕─┼─
//            │
// q[2]: ─────⊕─

// Step-through for debugging or animation
Circuit c(2);
c.h(0).cx(0, 1);
for (int i = 0; i <= c.n_gates(); ++i) {
    auto sv_i = c.statevector_at(i);   // state after i gates
    print_state(sv_i, c.n_qubits());
}

// Factory helpers
auto bell  = bell_pair();              // 2-qubit Bell state circuit
auto ghz   = ghz_state(5);            // 5-qubit GHZ
auto qft   = qft_circuit(4);          // 4-qubit QFT
qft.print();

// Custom gate injection
Gate my_gate = gate_rx(0.3);
Circuit(1).gate(my_gate, 0, "Rx").print();

// QFT on |001⟩
const real pi = 3.14159265358979;
Circuit qft3(3);
qft3.x(0)
    .h(2).cp(pi/2, 1, 2).cp(pi/4, 0, 2)
    .h(1).cp(pi/2, 0, 1)
    .h(0).swap(0, 2);
print_state(qft3.statevector(), 3);
// |000⟩  +0.3536+0.0000i  (12.5%)
// |001⟩  +0.2500+0.2500i  (12.5%)
// ...
```

---

## Visualisation

The window is divided into three regions:

**Circuit panel (left):** Qubit wires are drawn as horizontal lines. Each gate occupies a time column determined by a greedy layout pass that avoids qubit conflicts. Gate colours encode type:

| Colour | Gates |
|--------|-------|
| UCSB Gold | $H$ |
| Red | $X$, CNOT target $\oplus$ |
| Green | $Y$ |
| Blue | $Z$ |
| Orange | $S$, $T$ and their conjugates |
| Purple | $R_x$, $R_y$, $R_z$, SWAP |
| Teal | Controlled gates (CX, CY, CZ, CP, CCX) |

Gates already applied are drawn at full brightness; future gates are dimmed. The **currently applied gate** is highlighted with a white border.

**Histogram panel (right):** Vertical bars show $|\alpha_k|^2$ for every basis state $|k\rangle$. Bars exceeding 40% are gold; others are teal. Probability labels appear above bars above 2%.

**Stats strip (bottom):** Shows the most probable outcome, entanglement entropy $S(q_0)$, and statevector norm (a sanity check on unitarity). The three largest non-negligible amplitudes are displayed with real and imaginary parts.

---

## Controls

| Key | Action |
|-----|--------|
| **→** or **L** | Apply next gate (step forward) |
| **←** or **H** | Undo last gate (step back) |
| **SPACE** | Toggle auto-step (one gate per 0.55 s) |
| **R** | Reset to $|0\cdots0\rangle$ |
| **1** | Load Bell State |
| **2** | Load GHZ State |
| **3** | Load Grover Search |
| **4** | Load Quantum Teleportation |
| **5** | Load QFT₃ |

The step counter (top right) shows `STEP k / d` where $d$ is the circuit depth.

---

## Build

### Using CMake presets (recommended)

```bash
cmake --preset app-quantum       # configure: Release, quantum_demo only
cmake --build --preset app-quantum
./build/apps/quantum_demo/quantum_demo
```

`app-quantum` is one of several single-app presets defined in `CMakePresets.json`. Each preset sets `CMAKE_BUILD_TYPE`, enables exactly one app target, and disables everything else so the build stays fast. Available app presets:

| Preset | App | Description |
|--------|-----|-------------|
| `app-quantum` | `quantum_demo` | This app — quantum circuit simulator |
| `app-tdse`    | `tdse`         | 2D time-dependent Schrödinger equation |
| `app-fluid`   | `fluid_sim`    | 2D SPH fluid simulation |
| `app-fluid3d` | `fluid_sim_3d` | 3D SPH fluid simulation |
| `app-em`      | `em_demo`      | Electromagnetic field solver |
| `app-ising`   | `ising_sim`    | Ising model Monte Carlo |
| `app-ns`      | `ns_demo`      | 2D Navier-Stokes stress test |

To build every app at once:

```bash
cmake --preset apps
cmake --build --preset apps
```

### Manual configuration

```bash
cmake -B build -DNUMERICS_BUILD_QUANTUM_DEMO=ON
cmake --build build --target quantum_demo
./build/apps/quantum_demo/quantum_demo
```

---

## A Note on Quantum Error Correction

The current simulator is **ideal** — gates are perfectly unitary and measurement statistics follow the exact Born distribution. Adding realistic noise requires two extensions:

**1. Noise channels.** A depolarizing channel applies a random Pauli error after each gate with probability $p$:

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

This forces shot-by-shot simulation (one fresh run per shot). Adding `apply_depolarizing(sv, n, q, p, rng)` is straightforward; the cost is $O(\text{depth} \cdot 2^n \cdot \text{shots})$ instead of $O(\text{depth} \cdot 2^n + \text{shots})$.

**2. Mid-circuit measurement.** QEC syndrome extraction requires measuring ancilla qubits and applying conditional correction gates in the same circuit. This needs `measure(qubit) → int` which projects the statevector onto the $|0\rangle$ or $|1\rangle$ subspace of that qubit, renormalises, and returns the classical bit. The `Circuit` class would then need conditional branching (`if_bit`).

**3. Density matrices (for mixed states).** If decoherence accumulates without measurement, the system is in a *mixed state* described by a $2^n \times 2^n$ density matrix $\rho$ rather than a statevector. This costs $4^n$ memory — feasible only for $n \lesssim 12$.

A pragmatic alternative is the **stabilizer formalism** (Aaronson–Gottesman 2004): Clifford circuits ($H, S, \text{CX}$) plus Pauli measurements can be simulated in $O(n^2)$ time and space using a $2n \times 2n$ binary tableau, enabling large-scale QEC simulation. This is the approach taken by Stim, the fastest known QEC simulator.

---

## References

- M. A. Nielsen & I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press (2000) — comprehensive reference for all circuits
- L. K. Grover, *A fast quantum mechanical algorithm for database search*, STOC (1996)
- P. W. Shor, *Algorithms for quantum computation: discrete logarithms and factoring*, FOCS (1994) — QFT as a subroutine for factoring
- C. H. Bennett et al., *Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels*, PRL **70** (1993)
- D. M. Greenberger, M. A. Horne & A. Zeilinger, *Going beyond Bell's theorem*, in Bell's Theorem (1989)
- S. Aaronson & D. Gottesman, *Improved simulation of stabilizer circuits*, PRA **70** (2004) — efficient Clifford simulation
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum **5** (2021)
