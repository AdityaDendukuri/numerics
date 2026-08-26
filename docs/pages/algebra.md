# Algebraic Structure {#page_algebra}

The `algebra` module provides the scalar field, the vector space hierarchy, the property hierarchy carried by matrices and operators, and the runtime samplers that check the laws all of them claim.

Every numerical module is written against these. `linear` constrains solvers on them, `quadrature` takes a `ScalarFunction`, and `spatial` templates its coordinates on `Field`.

---

## 1. Scalar Fields

A field is a scalar type supporting the four arithmetic operations over a floating-point base:

```cpp
static_assert(num::Field<double>);
static_assert(num::Field<float>);
static_assert(num::Field<std::complex<double>>);
static_assert(!num::Field<int>); // Integers form a ring, not a field.
```

Field-generic helpers work for real and complex alike:

```cpp
num::scalars::conj(z);   // Complex conjugate; the identity on a real field.
num::scalars::re(z);     // Real part.
num::scalars::mag(z);    // Modulus |z|.
num::scalars::eps<T>();  // Machine epsilon of the underlying real field.
```

---

## 2. The Vector Space Hierarchy

Each concept adds one requirement to the one above it:

| Concept | Adds |
| :--- | :--- |
| `num::AdditiveGroup<V>` | Closure under addition, a zero, and inverses |
| `num::VectorSpace<V>` | Scalar multiplication compatible with the field |
| `num::InnerProductSpace<V>` | An inner product \f$\langle x,y \rangle\f$ |
| `num::NormedSpace<V>` | A norm \f$\|x\|\f$ |
| `num::HilbertSpace<V>` | Both, with \f$\|x\|^2 = \langle x,x \rangle\f$ |

```cpp
static_assert(num::VectorSpace<num::Vector>);
static_assert(num::VectorSpace<num::CVector>);
static_assert(num::VectorSpace<std::vector<float>>);       // Foreign container.
static_assert(num::HilbertSpace<num::Vector>);

static_assert(!num::VectorSpace<std::span<const double>>); // A view holds no sum.
static_assert(!num::VectorSpace<std::vector<int>>);        // Not a field.
```

Closure is the requirement indexing alone misses. A type must be able to construct an element and receive a sum, which is why a read-only view is rejected.

### Write code for any vector space

```cpp
template <num::VectorSpace V>
void normalize(V &v) {
    num::algebra::scale_inplace(v, num::scalar_t<V>(1) / num::algebra::norm_of(v));
}
```

The operations resolve to whichever form a type provides. `num::Vector` has free `dot` and `norm`, `num::SmallVec` has operators, and `std::vector<float>` has neither, so the contiguous kernel is used.

```cpp
num::algebra::inner(x, y);        // <x, y>, conjugating on a complex field.
num::algebra::norm_of(x);         // ||x||.
num::algebra::axpy_into(a, x, y); // y <- y + a x.
num::algebra::scale_inplace(v, a);// v <- a v.
num::algebra::zero<V>(n);         // The additive identity of dimension n.
```

---

## 3. Verifying the Laws

Structure is decided by the compiler. Whether a type obeys the axioms is not, so it is sampled:

```cpp
num::debug::verify_additive_group_axioms<num::Vector>(64); // Associativity, commutativity,
                                                           // identity, inverses.
num::debug::verify_vector_space_axioms<num::Vector>(64);   // Distributivity, scalar action.
num::debug::verify_inner_product_axioms<num::CVector>(64); // Conjugate symmetry, linearity,
                                                           // positive definiteness.
num::debug::verify_norm_axioms<num::Vector>(64);           // Homogeneity, triangle inequality,
                                                           // ||x||^2 == <x,x>.
num::debug::verify_hilbert_space_axioms<num::Vector>(64);  // All of the above.
```

The probes route through the type's own operations, so checking `num::Vector` checks the shipped `dot`, `norm`, `axpy`, and `scale` across whichever backend the build selected. Conjugate symmetry \f$\langle x,y \rangle = \overline{\langle y,x \rangle}\f$ is invisible on real data and corrupts every complex Krylov method, which is why it is checked rather than assumed.

---

## 4. The Property Hierarchy

Properties of a linear map form an inheritance hierarchy. A type declares one and satisfies every weaker one:

\f[
\text{linear} \subset \text{normal} \subset
\begin{cases}
\text{self\_adjoint} \subset \text{psd} \subset \text{spd} \\
\text{skew\_adjoint} \\
\text{unitary}
\end{cases}
\f]

```cpp
struct MyOperator {
    using properties = num::property::spd; // One declaration.

    num::idx rows() const;
    num::idx cols() const;
    void apply(const num::Vector &x, num::Vector &y) const;
};

static_assert(num::SPDOperator<MyOperator>);
static_assert(num::SelfAdjointOperator<MyOperator>); // Implied.
static_assert(num::NormalOperator<MyOperator>);      // Implied.
```

See @ref page_concepts for how the hierarchy gates solvers and how the claims are checked.
