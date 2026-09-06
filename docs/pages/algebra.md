# Algebraic Structure {#page_algebra}

scalar fields, vector spaces, generic vector algorithms, and the linear operator property hierarchy.

---

## 1. Scalar Fields (`num::field`)

Scalars supporting `+`, `-`, `*`, `/` over a floating-point base (`double`, `float`, `std::complex<double>`).

```cpp
#include <numerics.hpp>

static_assert(num::field<double>);
static_assert(num::field<float>);
static_assert(num::field<std::complex<double>>);
static_assert(!num::field<int>); // Integers form a ring, not a field
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
| `num::additive_group<V>` | Additive closure, identity zero, and inverses |
| `num::vector_space<V>` | Compatible scalar multiplication |
| `num::inner_product_space<V>` | Inner product \f$\langle x, y \rangle\f$ with conjugate symmetry |
| `num::normed_space<V>` | Norm \f$\Vert x \Vert\f$ satisfying homogeneity and triangle inequality |
| `num::hilbert_space<V>` | Inner product + norm with \f$\Vert x \Vert^2 = \langle x, x \rangle\f$ |

```cpp
static_assert(num::vector_space<num::vec>);
static_assert(num::vector_space<num::cvec>);
static_assert(num::vector_space<std::vector<float>>);       // Foreign container
static_assert(num::hilbert_space<num::vec>);
```

### Generic Vector Space Algorithms

```cpp
template <num::vector_space V>
void normalize(V& v) {
    num::algebra::scale_inplace(v, num::scalar_t<V>(1) / num::algebra::norm_of(v));
}
```

```cpp
num::math::inner(x, y);          // <x, y> (conjugating for complex field)
num::math::norm(x);              // ||x||
num::math::axpy(a, x, y);        // y <- y + a * x
num::math::scale(a, v);          // v <- a * v
num::math::zero_like(v);         // Additive zero of the same dimension
```

---

## 3. Axiom Verification

```cpp
num::debug::verify_additive_group_axioms<num::vec>(64);
num::debug::verify_vector_space_axioms<num::vec>(64);
num::debug::verify_inner_product_axioms<num::cvec>(64);
num::debug::verify_norm_axioms<num::vec>(64);
num::debug::verify_hilbert_space_axioms<num::vec>(64);
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
    using math_laws = num::math::type_list<num::law::spd>;
    using domain_type = num::vec;
    using codomain_type = num::vec;

    num::idx rows() const;
    num::idx cols() const;
    void apply(const num::vec& x, num::vec& y) const;
};

static_assert(num::spd_operator<MyOperator>);
static_assert(num::self_adjoint_operator<MyOperator>); // Implied
static_assert(num::normal_operator<MyOperator>);      // Implied
```

