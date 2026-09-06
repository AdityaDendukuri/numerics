# Mathematical Core Refactor Roadmap {#page_refactor_roadmap}

## Status

This document is the design target for the next architecture of Numerics. The
first migration slice is now implemented: the dependency-free `core/math`
protocol, immutable evidence with provenance, native and foreign vector models,
the PDE-to-CG invariant path, and the evidence-constrained CG, PCG, MINRES, and
GMRES family. All four algorithms use the same operations vocabulary and accept
foreign certified vector/operator models. Sections describing other domains
remain prospective until their own vertical slices are implemented.

Restricted-space evidence is also implemented for the first important singular
case. `positive_definite_on<Subspace>` is distinct from global positive
definiteness, `space::zero_sum` models the compatible subspace of a connected
graph Laplacian, and restricted PCG enforces membership and preservation through
the recurrence. This removes the former need to describe a singular Laplacian or
ApproxChol factor as globally SPD. The projected-operator adapter makes gauge
selection explicit when an approximate inverse returns a representative outside
the chosen subspace, and `pcg_on_method` carries the same restriction through the
problem-level solve interface.

The old property wrappers remain compatibility carriers while callers migrate.
They are immutable when they carry mathematical axioms, record provenance, and
feed the same canonical Krylov implementations; they are no longer a parallel
solver architecture.

The first dependency split is also implemented. `numerics::kernel` and
`numerics::core` are independently exported, dependency-free targets. BLAS,
LAPACK, OpenMP, FFTW, SuiteSparse, and SIMD configuration lives on named
capability targets; MPI and CUDA configure only their compiled capability
targets. `numerics::numerics` deliberately remains the compatibility umbrella
that selects the available host capabilities. Splitting portable containers and
compiled algorithms from that umbrella remains future work.

The refactor is guided by one sentence:

> Numerics is an open generic numerical system in which concepts name
> mathematical and computational structures, evidence-bearing values carry
> runtime invariants, and algorithms preserve that knowledge down to raw
> kernels.

The intended user experience combines three qualities:

- **NumPy-like immediacy:** useful concrete arrays, familiar free functions, and
  short calls for ordinary numerical work.
- **Julia-like genericity:** algorithms dispatch on capabilities and laws rather
  than on the library's concrete containers.
- **Strict mathematical meaning:** algorithms cannot consume values that lack
  evidence for required algebraic axioms or discrete structural invariants.

This is a refactor of the abstraction system, not a rewrite of proven numerical
kernels.

---

## 1. Architectural Constitution

### 1.1 Kernel computes

`kernel` contains raw computation over pointers, spans, dimensions, and
callables. It knows no matrices, graphs, differential equations, mathematical
property wrappers, diagnostic presets, or symbolic expressions.

Kernel requirements are mechanical: valid buffers, sizes, strides, workspace,
and callable signatures. A kernel may report numerical breakdown, but it does
not decide whether an input models the mathematical object required by a public
algorithm.

### 1.2 Core gives computation mathematical meaning

`core` owns the stable vocabulary shared across numerical domains:

- scalar algebra;
- spaces and their operations;
- maps between spaces;
- laws and axioms;
- evidence for properties of runtime values;
- dimensions and shapes;
- finite sets, relations, and other foundational discrete structures;
- certified theoretical metadata.

Core must remain small in implementation even though it is foundational in
meaning. A concept belongs in core only when multiple domains use it or when it
is needed to state a cross-domain invariant.

### 1.3 Domains refine core

Each numerical domain defines only the concepts it introduces, as refinements of
the core vocabulary:

```text
kernel                 raw computation
   ^
core/math              scalars, spaces, maps, axioms, evidence
core/discrete          sets, relations, incidence, ordering
core/theory            complexity claims, certificates, reductions
   ^
   +-- linear          matrices, factorizations, Krylov problems
   +-- ode             vector fields, flows, Hamiltonian systems
   +-- pde             discrete fields, differential operators, conservation
   +-- spectral        transforms and spectral structure
   +-- stochastic      measures, distributions, Markov kernels
   +-- stats           estimators and composable accumulators
   +-- structures      concrete graphs, queues, union-find, traversal
   +-- symbolic        expressions, theorem rules, lowering
```

In C++, refinement is expressed by conjunction. Concepts do not literally
inherit, but every domain concept contains the core concepts on which its
mathematical definition depends.

### 1.4 Algorithms require the weakest sufficient proposition

An algorithm must not require a stronger structure than its derivation uses:

- adaptive ODE error control needs a normed state space, not necessarily an
  inner-product space;
- GMRES needs a linear operator on an inner-product space;
- CG needs self-adjoint positive-definite action;
- Dijkstra needs ordered, nonnegative edge weights;
- BFS needs finite incidence, not weighted storage;
- a dense Cholesky backend needs both SPD evidence and dense representation.

### 1.5 Invariants are preserved, not repeatedly rediscovered

Construction and transformation propagate known facts. If a PDE discretization
mathematically produces an SPD operator, the returned type carries that evidence
into CG. The user should not need to restate the same theorem at every layer.

---

## 2. Four Different Kinds of Constraint

The current design sometimes places different claims under one concept or one
diagnostic policy. The new design separates four categories.

### 2.1 Computational protocols

Protocols are facts the compiler can determine from expressions:

```cpp
template<class V>
concept CoordinateReadable = requires(const V& v, index_t<V> i) {
    { v.size() };
    v[i];
};

template<class Op, class X, class Y>
concept OperatorProtocol = requires(const Op& op, const X& x, Y& y) {
    { op.domain_dimension() };
    { op.codomain_dimension() };
    op.apply(x, y);
};
```

Protocols say what a program can do. They do not assert algebraic laws.

### 2.2 Type-level mathematical models

A type models an algebraic structure when it supplies the protocol and explicitly
certifies the laws:

```cpp
template<class T, class Law>
inline constexpr bool models = false; // customization point

template<class V>
concept vector_space =
    VectorProtocol<V> && Models<V, law::vector_space>;
```

Shipped types provide certifications. Foreign types can opt in without deriving
from library classes. Law suites audit those certifications in tests.

This is strict enforcement of responsibility, not a claim that the compiler can
prove associativity or distributivity.

### 2.3 Value-level invariants

Properties such as positive definiteness, graph connectivity, sortedness, and a
valid probability mass are facts about individual values. They are carried as
immutable evidence:

```cpp
template<class Proposition, class T, class EvidenceKind>
class certified_ref;

template<class T, class P>
concept Carries = /* T contains evidence for proposition P */;
```

Evidence has three explicit origins:

```cpp
auto a = assume<axiom::spd>(A);  // caller supplies the proof
auto p = probe<axiom::spd>(A);   // numerical sampling; evidence, not proof
auto v = require<axiom::spd>(A); // exhaustive validation or failure
```

Construction can provide a fourth, stronger origin:

```cpp
// PROPOSED API -- part of the design target, not yet implemented.
auto L = pde::negative_dirichlet_laplacian(grid);
// The construction returns a certified self-adjoint negative-definite operator.
```

### 2.4 Theoretical certifications

Worst-case complexity, membership in a complexity class, NP-hardness, and
NP-completeness are properties of problem or algorithm families under a stated
input encoding and computational model. They cannot be inferred from a callable's
syntax and should not be presented as runtime validation.

They are first-class, curated certifications with provenance:

```cpp
template<class Algorithm>
struct complexity_certificate;

template<class Problem>
struct complexity_class;

template<class From, class To>
concept PolynomialReduction = /* certified reduction object */;
```

A certification records assumptions such as comparison model, RAM model, sparse
input encoding, randomized expected time, or amortized analysis. Its purpose is
to support documentation, algorithm selection, static inspection, and regression
tests without pretending C++ has proved the theorem.

---

## 3. Foundational Mathematical Vocabulary

The initial vocabulary should be deliberately smaller than a complete algebra
textbook. A new abstraction is added only when it constrains an algorithm,
selects an overload, enables invariant propagation, or supports a meaningful law
suite.

### 3.1 scalar hierarchy

Candidate initial hierarchy:

```text
Semiring
  Ring
    field
      RealField
      ComplexField

OrderedScalar
NormedScalar
```

Orthogonal refinements are preferable to forcing every property into one
inheritance tree. Ordering and norm are independent of the basic arithmetic
hierarchy.

### 3.2 Space hierarchy

```text
additive_group
  Module
    vector_space
      normed_space
      inner_product_space
        FiniteHilbertSpace
```

`FiniteHilbertSpace` explicitly names the computational setting. It does not
claim that a compiler has proved analytic completeness.

Associated types and operations include:

- `scalar_t<V>`;
- `dimension(v)`;
- `zero_like(v)` rather than requiring construction from an integer;
- `add`, `scale`, and `axpy`;
- `inner` and `norm` where appropriate;
- optional coordinate and contiguous-storage access as representation concepts.

The mathematical structure must not require a particular representation.

### 3.3 Maps and operators

```text
Map<X,Y>
  endomorphism<X>
  linear_map<X,Y>
    linear_operator<X,Y>
    AdjointableOperator<X,Y>
```

`OperatorProtocol` is callable structure. `linear_map` additionally requires a
type-level law certification. Properties of a particular map are evidence:

```text
self_adjoint
positive
positive_semidefinite
positive_definite
unitary
projection
mass_preserving
symplectic
```

Only properties with consumers or propagation rules enter the public vocabulary.

### 3.4 Foundational discrete vocabulary

```text
FiniteSet
Relation
equivalence_relation
PartialOrder
incidence_structure
FiniteGraph
```

Domain refinements then express directedness, simplicity, acyclicity,
connectivity, planarity, bipartiteness, and weight restrictions as type models or
value evidence depending on whether every value of the type has the property.

---

## 4. Evidence Semantics

Evidence is the heart of strict runtime invariants and must obey several rules.

### 4.1 Evidence views are non-owning and immutable

Validating an lvalue must not copy a dense matrix. A certified reference holds a
`const` reference or pointer and cannot expose arbitrary mutation. Mutation would
invalidate the proposition.

```cpp
auto P = require<axiom::spd>(A);
cholesky(P);       // accepted
// P(0, 0) = -1;  // ill-formed
```

Invariant-preserving transformations may return new evidence. Arbitrary mutation
returns to the uncertified base type.

### 4.2 Decidable prerequisites are always checked

An SPD assertion may take definiteness on faith, but it may not certify a
rectangular object. Shape compatibility, valid indices, buffer bounds, and
representation integrity are structural safety conditions, not optional
diagnostics.

### 4.3 Implication is explicit and minimal

The initial linear-operator implication chain is:

```text
positive_definite
  -> positive_semidefinite
    -> self_adjoint
      -> linear
```

Independent facts use a property set rather than artificial inheritance. A graph
can be independently connected, planar, simple, and nonnegative-weighted.

### 4.4 Evidence provenance is inspectable

Diagnostics may report whether a proposition was constructed, verified, probed,
or assumed. Algorithm validity depends on the proposition, while users and tests
can impose stronger provenance requirements where needed.

### 4.5 No global switch changes truth

Global presets must not make a certified value mean something different, disable
memory-safety checks, or cause `assume` to become `probe`. Expensive diagnostics
are explicit operations. Internal assertions may follow build policy, but public
invariant semantics remain stable.

---

## 5. Invariant Propagation Across Domains

### 5.1 Linear algebra

```cpp
template<spd_operator A, FiniteHilbertSpace V>
solver_result cg(const A& op, const V& b, V& x, cg_options = {});

template<dense_row_major A>
requires Carries<A, axiom::spd>
auto cholesky(const A& matrix);
```

The first signature depends on mathematical action. The second additionally
depends on stored representation.

### 5.2 ODEs

```cpp
template<class P>
concept ode_problem =
    normed_space<state_t<P>> &&
    VectorField<rhs_t<P>, state_t<P>> &&
    InitialValueProblemProtocol<P>;
```

Adaptive error estimation requires a normed state space. Hamiltonian problems
refine ODE problems with a symplectic state space and Hamiltonian evidence.
Symplectic steppers state the structure they preserve.

### 5.3 PDEs

Discrete function spaces and operators refine core spaces and maps. Builders
carry theorems from the discretization into their result types:

```cpp
// PROPOSED API -- part of the design target, not yet implemented.
auto L = pde::dirichlet_laplacian(grid);
// L: self-adjoint negative-definite operator

auto A = identity_like(L) - dt * L;
// Given dt > 0, symbolic/property propagation yields SPD evidence.

auto result = cg(A, rhs, solution);
```

This is the principal end-to-end acceptance test for the architecture.

### 5.4 Spectral methods

A transform is a linear operator between inner-product spaces. A normalized FFT
can carry unitary evidence; composition and adjoint preserve it. Parseval tests
audit the implementation but do not define the concept by themselves.

### 5.5 Stochastic systems

Foundational concepts include finite measures, probability measures,
distributions, and transition kernels. A Markov kernel carries positivity and
mass-preservation evidence. Finite-chain properties such as irreducibility,
aperiodicity, and reversibility are value-level propositions.

```cpp
template<class K>
concept MarkovKernel =
    PositiveOperator<K> && Carries<K, axiom::mass_preserving>;

template<class K>
concept ErgodicKernel =
    MarkovKernel<K> && Carries<K, axiom::ergodic>;
```

### 5.6 Discrete algorithms

```cpp
template<IncidenceGraph G>
auto bfs(const G& graph, vertex_t<G> source);

template<NonnegativeWeightedGraph G>
auto dijkstra(const G& graph, vertex_t<G> source);

template<ConnectedUndirectedGraph G>
auto minimum_spanning_tree(const G& graph);
```

Concrete graph implementations maintain permanent representation invariants.
Properties of an individual graph are constructed, verified, or assumed through
the common evidence system.

---

## 6. Complexity Theory and NP-Completeness

Complexity should be part of the abstract bridge, but with precise semantics.

### 6.1 Complexity belongs to algorithm families

A complexity certificate identifies:

- input size variables such as `V`, `E`, `n`, `nnz`, or condition-dependent
  iteration count;
- time and auxiliary-space bounds;
- worst-case, expected, amortized, or output-sensitive interpretation;
- computational model;
- input encoding and preconditions;
- reference or proof note.

Examples:

```text
BFS:       time O(V + E), space O(V), adjacency-incidence model
Dijkstra:  time O((V + E) log V), nonnegative ordered weights, binary heap
UnionFind: amortized O(alpha(n)), union by rank plus path compression
CG:        O(k * apply(A)) time and O(n) workspace; convergence depends on spectrum
```

Asymptotic certification must not obscure numerical complexity. Iterative
methods should state both per-iteration cost and convergence parameters.

### 6.2 Complexity classes belong to problems

The vocabulary may include:

```text
DecisionProblem
SearchProblem
OptimizationProblem
CertificateVerifier
PolynomialReduction

P
NP
coNP
NP_hard
NP_complete
```

An NP certification requires a polynomial-time verifier and polynomially bounded
certificate under a stated encoding. NP-completeness additionally requires
NP-hardness through a certified reduction chain.

The library will not claim to mechanically prove these facts. They are curated
theorem metadata that can be inspected and tested:

```cpp
static_assert(ComplexityClass<VertexCoverDecision> == theory::NP_complete);
static_assert(PolynomialVerifier<VertexCoverCertificate>);
```

### 6.3 Theoretical metadata should affect user choices

The information becomes valuable when it changes behavior:

- documentation can show tractable and intractable variants together;
- generic `solve` can require an explicit strategy for NP-hard problems;
- exact exponential algorithms can require a budget or bounded-size policy;
- approximation algorithms can expose approximation guarantees;
- reductions can generate test instances and cross-check certificates;
- benchmarks can test empirical scaling against the certified cost model.

No default API should silently choose an exponential exact algorithm for an
unbounded NP-hard problem.

### 6.4 Approximation and randomized guarantees

The theoretical layer should eventually represent:

- approximation ratios;
- additive versus multiplicative error;
- high-probability bounds;
- expected runtime;
- Monte Carlo versus Las Vegas algorithms;
- fixed-parameter tractability with an explicit parameter;
- condition-number-dependent numerical bounds.

These are propositions with assumptions, not marketing labels.

### 6.5 Parameterized algorithms are a useful second spine

Parameterized complexity is a particularly good fit for this architecture
because it can change admissibility and dispatch instead of merely decorating an
algorithm.  A parameterized problem has an instance `x`, an explicit parameter
`k(x)`, and a stated encoding.  An implementation may then carry one of these
curated guarantees:

```text
FPT:        f(k) * n^O(1)
XP:         n^f(k)
kernel:     equivalent instance with size bounded by g(k)
exact:      exponential dependence and an explicit resource budget
```

Treewidth, pathwidth, branchwidth, solution size, rank, sparsity, and geometric
dimension are possible parameters; they are not interchangeable.  Each
parameter object must say how it is obtained, whether computing it is itself
hard, and whether it is an exact value, an upper bound, or a heuristic estimate.

A kernelization step is not just preprocessing.  It must return evidence that
the reduced instance is equivalent to the original decision problem, plus the
claimed size bound.  Dynamic programs over a decomposition must require a
validated decomposition rather than accepting an unstructured integer called
`treewidth`.  Dispatch may then select an FPT implementation only when the
required parameter and structural evidence are present.

This is where work in the style of modern parameterized algorithms is directly
useful.  The project should encode the interface between theorem and algorithm;
it should not attempt to turn the C++ type system into a proof assistant or
populate the core with the full parameterized-complexity hierarchy.

### 6.6 Computational geometry needs robust predicate semantics

Computational geometry is also valuable, and its most immediate contribution is
stronger than a complexity label: it forces the distinction between a
mathematical predicate and its floating-point realization.  A geometry module
should therefore refine the common spine with:

- dimension, coordinate field, metric, orientation, and topology as associated
  structure;
- explicit exact, adaptive, filtered, and approximate predicate policies;
- certified signs for predicates such as orientation and in-circle tests;
- a declared degeneracy policy rather than accidental tie-breaking;
- separation of combinatorial topology from coordinate storage;
- output-sensitive bounds that name both input and output size;
- construction results that retain the evidence needed by downstream meshes,
  spatial searches, and PDE discretizations.

For example, a triangulation algorithm should consume certified predicate
semantics and return a complex with incidence and orientation invariants.  A PDE
mesh should consume that result; it should not independently infer topology from
raw floating-point coordinates.  This gives computational geometry a natural
place downstream of core and upstream of spatial/PDE algorithms without making
geometry vocabulary part of every numerical type.

### 6.7 Admission rule for theoretical structure

A theoretical-CS abstraction enters the public spine only if at least one of the
following is true:

1. it rejects an invalid program or unsupported assumption;
2. it selects a materially different algorithm or backend;
3. it changes a returned correctness, approximation, probability, or complexity
   guarantee;
4. it carries evidence required by a downstream computation.

Named classes, theorem references, and asymptotic expressions that do none of
these belong in documentation or benchmark metadata.  This rule keeps the
theoretical layer deep without allowing it to become ornamental taxonomy.

---

## 7. Symbolic Layer

A symbolic layer is valuable if its purpose is invariant propagation and
lowering, not general computer algebra.

### 7.1 Initial scope

The first symbolic layer represents typed expressions over operators, scalars,
dimensions, and costs:

```text
identity(A)
transpose(A)
adjoint(A)
scale(a, A)
sum(A, B)
compose(A, B)
product(A, B)
kronecker(A, B)
block(A, B, C, D)
inverse(A)          only as a formal expression with explicit prerequisites
```

Every expression carries:

- domain and codomain;
- scalar field;
- shape constraints;
- known propositions;
- storage/materialization status;
- a symbolic cost expression.

### 7.2 Theorem rules

Property propagation is an explicit, inspectable rule registry. Candidate rules
include:

```text
unitary(A)                  -> unitary(adjoint(A))
unitary(A) and unitary(B)   -> unitary(compose(A, B))
self_adjoint(A)             -> self_adjoint(a * A) when a is real
PSD(A) and PSD(B)           -> PSD(A + B)
SPD(A) and PSD(B)           -> SPD(A + B)
linear(A) and linear(B)     -> linear(compose(A, B))
full_column_rank(A)         -> SPD(adjoint(A) * A)
Markov(A) and Markov(B)     -> Markov(compose(A, B))
symplectic(A), symplectic(B)-> symplectic(compose(A, B))
```

Rules must list side conditions. The system must refuse to propagate a property
when a sign, rank, commutation, boundary-condition, or domain compatibility
condition is unknown.

### 7.3 Lowering

Symbolic expressions lower to one of:

- a lazy callable operator;
- a materialized dense, sparse, banded, or block representation;
- a fused raw kernel;
- a domain-specific solver plan.

Lowering uses mathematical properties and representation concepts without
changing their meaning. For example, a Kronecker sum may remain matrix-free for
CG but materialize for a small direct solve.

### 7.4 Complexity algebra

The symbolic layer can represent cost expressions such as:

```text
apply(A + B)       = apply(A) + apply(B) + O(n)
apply(A compose B) = apply(B) + apply(A)
CG(A)              = k * apply(A) + O(k*n)
```

Simplification remains conservative. Cost metadata should help compare plans,
not pretend that asymptotic notation predicts hardware performance.

### 7.5 Explicit non-goals for the first implementation

The initial symbolic layer will not attempt:

- arbitrary symbolic integration;
- general equation solving;
- unrestricted expression rewriting;
- automated theorem proving;
- full symbolic differentiation;
- a replacement for SymPy, Mathematica, or a compiler optimizer.

The smallest useful milestone is typed operator expressions with shape checking,
property propagation, complexity annotations, and lowering to existing kernels.

---

## 8. Public API Shape

The direct numerical API remains canonical:

```cpp
dot(x, y);
matmul(A, B, C);
cg(A, b, x, options);
rk45(problem, options);
fft(plan, input, output);
```

Concrete defaults keep ordinary calls short. Generic algorithms remain open to
foreign types through protocol and model customization.

Strict calls read naturally:

```cpp
cg(require<axiom::spd>(A), b, x);
cholesky(assume<axiom::spd>(A));
dijkstra(require<axiom::nonnegative_weights>(G), source);
```

Domain constructors should usually eliminate this ceremony:

```cpp
auto A = pde::backward_euler_operator(grid, diffusivity, dt);
auto result = cg(A, rhs, solution); // A already carries SPD evidence.
```

`solve(problem, algorithm)` remains an optional generic façade. It must be an
open customization mechanism rather than a central table enumerating every
supported pair. Direct algorithms remain available and define the underlying
semantics.

---

## 9. Proposed Source Layout

The exact filenames are provisional:

```text
include/
  kernel/                    unchanged raw computation

  core/
    types.hpp
    math/
      models.hpp             Models<T, Law> customization
      scalar.hpp             scalar protocols and laws
      space.hpp              vector, normed, inner-product spaces
      map.hpp                maps and operator protocols
      axioms.hpp             shared value propositions
      evidence.hpp           assume/probe/require and certified_ref
      laws.hpp               reusable law-test declarations
    discrete/
      set.hpp
      relation.hpp
      incidence.hpp
    theory/
      complexity.hpp
      problem.hpp
      certificate.hpp
      reduction.hpp

  container/                 concrete storage and representation concepts
  symbolic/                  expression nodes, theorem rules, lowering
  linear/                    domain refinements and algorithms
  ode/
  pde/
  spectral/
  stochastic/
  stats/
  structures/
```

The public namespace may remain mostly flat for usability even when definitions
are physically organized. Internal taxonomy should not force verbose user calls.

---

## 10. Dependency and Backend Constitution

Numerics is primarily a template and header-based mathematical library. External
libraries are narrow implementations of leaf operations; they do not define an
alternative object model, concept system, or public semantics.

The central rule is:

> A backend may change how a valid operation is computed. It may not change what
> the operation means, which prerequisites it requires, or which invariants its
> result carries.

### 10.1 Header-first does not mean header-only at any cost

The following belong in headers:

- mathematical protocols, models, and concepts;
- evidence and proposition types;
- generic algorithms over spaces and maps;
- symbolic expressions and theorem rules;
- sequential portable kernels;
- representation-based dispatch;
- small backend adapters whose vendor types do not leak into core APIs.

Compiled translation units remain appropriate for narrow ABI islands:

- FFTW plan ownership;
- SuiteSparse KLU and UMFPACK handles;
- CUDA kernels and runtime calls;
- MPI bindings;
- vendor APIs whose headers or macros should not reach every consumer;
- optional explicit instantiations that reduce build time without defining the
  generic implementation.

The template implementation is authoritative. A compiled backend is an optional
model of the same leaf-operation contract.

### 10.2 Dependencies are capabilities, not global configuration

Optional dependencies must not attach themselves transitively to the raw kernel
or mathematical core. Finding MPI must not make every `vec` consumer include
MPI flags; finding CUDA must not change the semantics of a host container; finding
BLAS must not force CBLAS headers and link flags into a project that only uses a
root finder.

The target structure should converge toward:

```text
numerics::kernel             standard-library-only raw headers
numerics::core               mathematical/discrete/theory headers
numerics::containers         concrete host representations
numerics::algorithms         generic numerical headers

numerics::backend::blas      opt-in BLAS leaf implementations
numerics::backend::lapack    opt-in LAPACK factorizations
numerics::backend::openmp    opt-in parallel leaf implementations
numerics::backend::fftw      opt-in FFTW plan implementation
numerics::backend::suitesparse
numerics::backend::cuda
numerics::backend::mpi
numerics::io::json
```

An umbrella target may select a convenient set, but the foundational targets
remain dependency-free and independently installable. Package configuration
re-finds only dependencies belonging to targets a consumer requests.

The current implementation provides `kernel`, `core`, individual host backend
components, and the compatibility `numerics`/`backends` components. An installed
consumer can request only the foundation without triggering optional discovery:

```cmake
find_package(numerics REQUIRED COMPONENTS core)
target_link_libraries(my_target PRIVATE numerics::core)
```

Configuration-time assertions and an installed-package consumer test enforce
that this path works even when discovery of BLAS, LAPACK, OpenMP, MPI, FFTW,
PkgConfig, and CUDA is explicitly disabled.

Capability macros and compiler flags must be target-local. They must not change
the definition or layout of core public types across translation units.

### 10.3 Backend coverage is intentionally narrow

Generic algorithms can support a broad mathematical space while a vendor backend
supports a small intersection:

```cpp
template<class V>
concept BlasVector =
    contiguous_vector<V> &&
    BlasScalar<scalar_t<V>> &&
    HostAccessible<V>;

template<class M>
concept LapackMatrix =
    DenseStridedMatrix<M> &&
    LapackScalar<scalar_t<M>> &&
    HostAccessible<M>;
```

BLAS need not support every `vector_space`; CUDA need not support every operator;
FFTW need not support every transform scalar. Unsupported combinations use the
generic implementation or fail capability selection without weakening the
mathematical API.

This small coverage surface is a benefit: each external binding can be audited
against a precise leaf contract.

### 10.4 Safe computation has one semantic entrance

Every normal public computation is constrained by the core rules before backend
selection:

```text
user/domain value
  -> protocol and model constraints
  -> value evidence and shape checks
  -> mathematical algorithm
  -> representation dispatch
  -> execution/backend selection
  -> raw or vendor leaf
```

A backend is never a second entrance around those checks. Vendor namespaces are
implementation detail or explicitly raw APIs. A call to BLAS, cuBLAS, FFTW, or
SuiteSparse occurs only after the same prerequisites required by the sequential
path have been established.

Concrete shipped types carry their certifications automatically, so strictness
does not make common calls verbose:

```cpp
dot(x, y);               // vec is already a certified inner-product space.
fft(plan, x, y);         // Plan and spaces establish transform compatibility.
cg(A, b, x);             // A already carries SPD evidence.
```

### 10.5 Mathematical properties and execution properties are distinct

Backend choice introduces computational facts that do not belong in the
mathematical property lattice:

```text
memory_space:       host, device, distributed
execution:          sequential, threaded, vectorized, device, collective
determinism:        deterministic, order-dependent, nondeterministic
reduction_model:    serial order, tree reduction, vendor-defined
precision:          scalar precision, accumulation precision, mixed precision
availability:       compile-time, runtime device, runtime communicator
```

These execution properties can constrain dispatch and numerical guarantees. They
must not be confused with facts such as linearity, self-adjointness, positivity,
or mass preservation.

Floating-point implementations approximate exact laws. Backend conformance means
agreement within a stated error model, not bitwise equality unless deterministic
reproducibility is explicitly required.

### 10.6 Backend selection must be explicit about failure and fallback

Automatic selection may choose the best available conforming implementation. An
explicit request for an unavailable backend should not silently run a different
backend. It should fail at compile time for a static request or report a clear
runtime capability error for a dynamic request.

```cpp
matmul(A, B, C);                  // automatic conforming selection
matmul(A, B, C, backend::blas);   // BLAS specifically required
```

Silent fallback makes performance, determinism, and memory-space reasoning
unreliable. Fallback belongs to an explicit policy such as `prefer_blas`, not to
the meaning of `backend::blas`.

### 10.7 Host, device, and distributed storage are real representations

CUDA and MPI should not be boolean modes hidden inside ordinary host containers.
They introduce different representation and operation requirements:

- a device vector has device-accessible storage and transfer semantics;
- a distributed vector has local storage, a global dimension, a partition, and
  collective inner products;
- a distributed operator must agree with the vector partition and communicator;
- a device operator must be applicable in the selected memory space.

These are representation concepts and associated types:

```text
HostVector
DeviceVector
DistributedVector
GloballyReducedInnerProductSpace
DeviceApplicableOperator
DistributedLinearOperator
```

The mathematical algorithm may remain generic, while its operations select the
appropriate local, device, or collective leaves. Core host types must not change
layout when CUDA or MPI is detected.

### 10.8 Special treatment by dependency

#### BLAS and LAPACK

BLAS supplies dense level-1/2/3 leaf operations for supported scalar and layout
concepts. LAPACK supplies selected factorization implementations. Vendor headers
and link dependencies should be isolated to their backend target when practical.
Integer-width, layout, transpose, conjugation, and aliasing conventions are part
of the adapter contract.

#### OpenMP

OpenMP is an opt-in execution implementation. Pragmas in templates are acceptable
only through a target that deliberately propagates the necessary compile and link
flags. Parallel reductions declare their determinism and numerical error model.

#### SIMD

Portable or intrinsic SIMD is a header-level leaf optimization. Architecture
flags must not globally raise the instruction-set requirement of unrelated
consumer code. Use scoped targets, function multiversioning, or compiled dispatch
where portability requires it.

#### FFTW

FFTW is a compiled plan backend behind the common transform contract. Transform
normalization, direction, shape, aliasing, and unitary scaling conventions are
defined by Numerics and adapted explicitly to FFTW.

#### SuiteSparse

KLU and UMFPACK are compiled sparse-factorization providers behind PIMPL or
equivalent narrow handles. CSR/CSC conversion, index width, ordering, singularity,
and factorization failure map to Numerics concepts and result types.

#### CUDA and cuBLAS

CUDA provides device storage and device leaf operations. cuBLAS may implement
BLAS-like contracts for device representations. Device availability is a runtime
capability; scalar, layout, stream, and memory-space compatibility remain
compile-time constraints where possible.

#### MPI

MPI supplies collective operations for distributed representations. MPI
initialization and communicator lifetime remain explicit runtime resources. A
serial build may provide separate serial types or adapters, but must not pretend
that an explicitly requested MPI execution occurred.

#### JSON and other I/O

Serialization is an optional boundary module. It validates representation and
invariants when reconstructing typed values; deserialization must not manufacture
mathematical evidence merely because a file contains a property label.

### 10.9 Backend conformance suite

Every backend implementation runs the same contract tests as the portable leaf:

- shape, stride, aliasing, and empty-input behavior;
- real and complex conventions;
- numerical agreement under a documented tolerance model;
- law probes appropriate to the operation;
- failure and unavailable-capability behavior;
- host/device/distributed residency transitions;
- deterministic behavior when claimed;
- sanitizer and race checks where supported.

Higher-level invariant tests run over backend matrices. CG, FFT, factorization,
ODE, PDE, and stochastic tests must not test only the sequential path. This is how
the invariant thread reaches every computation rather than ending at dispatch.

### 10.10 Explicit escape levels

Two escape mechanisms serve different purposes:

```cpp
assume<axiom::spd>(A)   // preserves the mathematical thread; caller supplies proof
unchecked::matmul(...)  // bypasses decidable structural checks
kernel::matmul(...)// raw buffers and caller-owned preconditions
```

`assume` is not an unsafe algorithm. It creates required evidence with explicit
caller responsibility. `unchecked` skips mechanically decidable validation and
is reserved for measured internal or expert use. `num::kernel` is the final raw
layer.

Normal downstream code uses neither `unchecked` nor `num::kernel` directly. All
three levels remain greppable and unambiguous.

---

## 11. Migration Plan

### Phase 0: Freeze the constitution and audit the current API

1. Approve the four constraint categories in section 2.
2. Inventory every current concept and classify it as protocol, type model,
   value proposition, representation, domain refinement, or theoretical claim.
3. Inventory every algorithm's actual mathematical and storage requirements.
4. Record current examples, compile times, runtime benchmarks, and binary sizes.
5. Mark documentation that describes aspirations rather than implemented
   enforcement.
6. Record every optional dependency, the targets and compile definitions it
   reaches, and every public type whose definition changes with availability.

Deliverable: a concept/algorithm matrix with no ambiguous entries.

### Phase 1: Implement the minimal mathematical core

1. Add `Models<T, Law>` and associated-type customization.
2. Define the scalar, space, and map protocols.
3. Define the minimal law hierarchy used by existing algorithms.
4. Certify shipped scalar and vector types.
5. Convert existing algebra diagnostics into reusable law suites.
6. Prove that one foreign vector type can opt in without inheritance.
7. Establish dependency-free `kernel` and `core` installation targets.

Deliverable: generic vector operations and one Krylov primitive compile against
both `num::vec` and a foreign certified vector type.

### Phase 2: Replace property wrappers with immutable evidence

1. Implement `certified_ref` and proposition sets.
2. Implement explicit `assume`, `probe`, and `require` paths.
3. Enforce decidable prerequisites regardless of diagnostic policy.
4. Remove mutable access through evidence-bearing references.
5. Add evidence provenance and implication tests.
6. Remove or deprecate global presets that change invariant semantics.

Deliverable: an SPD certificate cannot survive arbitrary mutation, cannot wrap a
rectangular matrix, and does not copy an lvalue.

### Phase 3: Migrate one complete linear/PDE vertical slice

Use the Dirichlet diffusion problem as the architectural test:

```text
grid
  -> discrete function space
  -> Laplacian construction
  -> self-adjoint/definiteness propagation
  -> backward-Euler operator
  -> SPD evidence
  -> generic CG
  -> raw kernel
```

This phase also:

1. makes CG generic over a certified finite Hilbert space;
2. separates mathematical operator constraints from storage fast paths;
3. removes the duplicate public `unsafe::cg` path in favor of explicit
   assumptions;
4. establishes compile-fail tests for missing evidence;
5. benchmarks abstraction overhead against the current implementation.
6. runs the same slice through the portable and one external leaf backend without
   changing its mathematical types or evidence.

Deliverable: one end-to-end example in which the user never manually retags a
property already established by construction.

### Phase 4: Migrate discrete structures and add theory metadata

1. Generalize graph algorithms over incidence concepts.
2. Separate permanent representation invariants from per-value graph properties.
3. Introduce directed, undirected, weighted, and nonnegative-weighted models.
4. Add initial complexity certificates for BFS, DFS, Dijkstra, union-find, and
   minimum spanning tree.
5. Introduce problem, verifier, and reduction protocols.
6. Add one carefully scoped NP-complete example, such as Vertex Cover decision,
   with a polynomial certificate verifier and documented reduction provenance.
7. Add one parameterized algorithm whose parameter changes dispatch, preferably
   a bounded-treewidth or solution-size example with explicit FPT cost.
8. Represent a reduction or kernelization result as evidence-bearing output, not
   as an undocumented preprocessing mutation.

Deliverable: theoretical metadata is inspectable and useful without being
misrepresented as compiler-proved mathematics.

Computational geometry follows this phase as its own vertical slice: robust
orientation predicates, one invariant-preserving planar structure, and one
consumer in spatial search or mesh construction.  It should reuse the common
evidence and cost protocols rather than expand `core` with geometry-specific
concepts.

### Phase 5: Prototype the symbolic operator layer

1. Implement identity, scale, sum, composition, and adjoint nodes.
2. Enforce domain, codomain, scalar, and shape compatibility.
3. Implement a small property-rule registry.
4. Propagate symbolic cost expressions.
5. Lower expressions to callable operators.
6. Add optional materialization for dense and sparse representations.

Deliverable: `I - dt*L` carries SPD evidence under explicit side conditions and
lowers to the existing matrix-free CG path.

### Phase 6: Extend the invariant thread domain by domain

Migrate in this order:

1. remaining linear solvers and factorizations;
2. spectral transforms;
3. ODE and Hamiltonian systems;
4. stochastic kernels and distributions;
5. remaining PDE and field abstractions;
6. statistics and spatial structures.

Each migration must begin with actual algorithm requirements and end with a
vertical example. No domain receives a large concept taxonomy in advance of
consumers.

### Phase 7: Consolidate the public surface

1. Remove transitional wrappers and duplicate overload families.
2. Make direct free functions the canonical API.
3. Make `solve` an open, optional façade.
4. Finish leaf-level backend dispatch and remove runtime backend parameters from
   mathematical algorithms where possible.
5. Rebuild the umbrella header from domain umbrellas without duplicate includes.
6. Rewrite user documentation around workflows, with the mathematical model
   available as a deeper guide.
7. split optional dependencies into opt-in exported targets and ensure package
   configuration discovers only requested capabilities.
8. remove backend macros that alter core public type definitions.

---

## 12. Compatibility Strategy

The migration should be incremental rather than a flag-day rewrite.

- Keep raw kernels stable unless a benchmark or correctness issue requires a
  change.
- Introduce new core concepts alongside existing ones during the vertical
  prototype.
- Provide forwarding aliases only when their semantics remain identical.
- Deprecate rather than alias names whose meaning changes; an alias must not make
  an old sampled assertion look like a new zero-cost assumption.
- Migrate examples early so the intended user surface remains visible.
- Remove transitional APIs at a declared major-version boundary.

Compatibility is subordinate to semantic honesty. Code that relied on mutable
certificates or disabled structural safety should fail loudly during migration.

---

## 13. Verification Strategy

### 13.1 Compile-time tests

- protocol satisfaction and rejection;
- explicit model opt-in;
- law implication and independent property composition;
- algorithm rejection when evidence is absent;
- domain/codomain and scalar incompatibility;
- immutability of certified references;
- foreign-type interoperability;
- symbolic rule side conditions.

Compile-fail tests are part of the public contract, not incidental diagnostics.

### 13.2 Runtime law and invariant tests

- scalar, vector-space, norm, and inner-product laws;
- linearity, adjoint, symmetry, and definiteness probes;
- graph relation and incidence laws;
- probability positivity and normalization;
- Markov mass preservation;
- symplectic and conservation diagnostics;
- verifier correctness for theoretical certificates.

Randomized tests use deterministic seeds and report counterexamples.

### 13.3 Construction and mutation tests

- constructors establish exactly their documented invariants;
- invariant-preserving operations retain evidence;
- arbitrary mutation discards or cannot access evidence;
- invalid shapes and indices are rejected independently of diagnostics;
- evidence never dangles and lvalue certification does not copy storage.

### 13.4 Numerical and performance tests

- convergence and backward-error tests remain authoritative;
- generic paths are compared with raw kernels;
- no abstraction allocation occurs inside iteration loops unless documented;
- compile-time and binary-size budgets are tracked;
- empirical scaling is compared with complexity metadata, with constants and
  hardware effects reported separately.

---

## 14. Decisions Required Before Phase 1

The following decisions should be made through prototypes rather than abstract
preference:

1. Whether law certification uses a variable-template customization point,
   traits, tag invocation, or a combination.
2. How associated scalar, state, domain, and codomain types are discovered.
3. Whether evidence stores one proposition or a normalized set of independent
   propositions.
4. Whether constructed, verified, probed, and assumed evidence share one runtime
   representation or distinct types.
5. How symbolic expressions own or reference operands safely.
6. Whether complexity expressions are compile-time types, runtime symbolic
   values, or a small hybrid.
7. Which existing APIs can migrate without a major-version break.
8. Whether each external backend is header-adapted or isolated behind a compiled
   ABI boundary.
9. Which execution properties participate in static dispatch and which are
   runtime resources.

The first prototype should compare alternatives on diagnostics, compile time,
generated code, foreign-type ergonomics, and conceptual clarity.

---

## 15. Definition of Success

The refactor succeeds when all of the following are true:

1. A beginner can perform ordinary array and numerical operations without
   learning the complete concept hierarchy.
2. An external type can participate by providing operations and explicit law
   certifications, without inheriting from library classes.
3. Algorithms reject missing mathematical or structural prerequisites at compile
   time when evidence is required.
4. Runtime value evidence cannot be invalidated while retaining its certified
   type.
5. Domain constructors propagate established invariants into downstream
   algorithms without repeated user assertions.
6. Mathematical algorithms are generic over spaces and maps; storage and backend
   specialization occur below them.
7. Complexity and NP-completeness claims state their model, assumptions, and
   provenance and influence meaningful choices.
8. Symbolic expressions propagate only justified facts and lower to the existing
   numerical kernels without mandatory runtime overhead.
9. The raw kernel remains independently usable and benchmark-equivalent.
10. The documentation explains one continuous abstraction system rather than
    separate concept frameworks for each domain.
11. A dependency-free consumer can use `kernel`, `core`, containers, and generic
    algorithms without discovering or linking vendor libraries.
12. Every enabled backend passes the same semantic, invariant, and numerical
    conformance suite as the portable implementation.
