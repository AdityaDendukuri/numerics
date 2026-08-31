# Algebraic Structure {#page_algebra}

Scalar fields, vector spaces, generic vector algorithms, and the linear operator property hierarchy.

---

## 1. Scalar Fields (num::Field)

Scalars supporting `+`, `-`, `*`, `/` over a floating-point base (`double`, `float`, `std::complex<double>`).

```cpp
#include <numerics.hpp>

static_assert(num::Field<double>);
static_assert(num::Field<float>);
static_assert(num::Field<std::complex<double>>);
static_assert(!num::Field<int>); // Integers form a ring, not a field
```

### Scalar Helpers
```cpp
num::scalars::conj(z);  // Complex conjugate; identity for real types
num::scalars::re(z);    // Real component
num::scalars::mag(z);   // Modulus |z|
num::scalars::eps<T>(); // Machine epsilon of underlying real field
```

---

## 2. Vector Spaces

| Concept | Structure |
| :--- | :--- |
| `num::AdditiveGroup<V>` | Additive closure, identity zero, and inverses |
| `num::VectorSpace<V>` | Compatible scalar multiplication |
| `num::InnerProductSpace<V>` | Inner product \f$\langle x, y \rangle\f$ with conjugate symmetry |
| `num::NormedSpace<V>` | Norm \f$\Vert x \Vert\f$ satisfying homogeneity and triangle inequality |
| `num::HilbertSpace<V>` | Inner product + norm with \f$\Vert x \Vert^2 = \langle x, x \rangle\f$ |

```cpp
static_assert(num::VectorSpace<num::Vector>);
static_assert(num::VectorSpace<num::CVector>);
static_assert(num::VectorSpace<std::vector<float>>);       // Foreign container
static_assert(num::HilbertSpace<num::Vector>);
static_assert(!num::VectorSpace<std::span<const double>>);// Views cannot receive sums
```

### Generic Vector Space Algorithms

```cpp
template <num::VectorSpace V>
void normalize(V& v) {
    num::algebra::scale_inplace(v, num::scalar_t<V>(1) / num::algebra::norm_of(v));
}
```

```cpp
num::algebra::inner(x, y);        // <x, y> (conjugating for complex field)
num::algebra::norm_of(x);         // ||x||
num::algebra::axpy_into(a, x, y); // y <- y + a * x
num::algebra::scale_inplace(v, a);// v <- a * v
num::algebra::zero<V>(n);         // Additive zero element of dimension n
```

---

## 3. Axiom Verification

```cpp
num::debug::verify_additive_group_axioms<num::Vector>(64);
num::debug::verify_vector_space_axioms<num::Vector>(64);
num::debug::verify_inner_product_axioms<num::CVector>(64);
num::debug::verify_norm_axioms<num::Vector>(64);
num::debug::verify_hilbert_space_axioms<num::Vector>(64);
```

---

## 4. Property Hierarchy

The properties of linear operators form an axiomatic hierarchy. Declaring a specialized property tag automatically satisfies all parent concepts:

\f[
\text{linear} \subset \text{normal} \subset
\begin{cases}
\text{self-adjoint} \subset \text{psd} \subset \text{spd} \\
\text{skew-adjoint} \\
\text{unitary}
\end{cases}
\f]

```cpp
struct MyOperator {
    using properties = num::property::spd;

    num::idx rows() const;
    num::idx cols() const;
    void apply(const num::Vector& x, num::Vector& y) const;
};

static_assert(num::SPDOperator<MyOperator>);
static_assert(num::SelfAdjointOperator<MyOperator>); // Implied
static_assert(num::NormalOperator<MyOperator>);      // Implied
```

