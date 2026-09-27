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
| `num::vector_space<V>` | `dimension`, `zero_like`, `scale` and `axpy` over a field |
| `num::inner_product_space<V>` | plus `inner` and its induced `norm` |

A space is decided by its operations, so a standard container qualifies with no declaration.

```cpp
static_assert(num::vector_space<num::vec>);
static_assert(num::vector_space<num::cvec>);
static_assert(num::vector_space<std::vector<float>>);       // foreign container
static_assert(num::inner_product_space<num::vec>);
```

### Generic Vector Space Algorithms

```cpp
template <num::inner_product_space V>
void normalize(V& v) {
    num::math::scale(num::scalar_t<V>(1) / num::math::norm(v), v);
}
```

```cpp
num::math::inner(x, y);          // <x, y> (conjugating for complex field)
num::math::norm(x);              // ||x||
num::math::axpy(a, x, y);        // y <- y + a * x
num::math::scale(a, v);          // v <- a * v
num::math::zero_like(v);         // additive zero of the same dimension
```

Each operation calls a type's `tag_invoke` overload when it has one, and otherwise loops over
`v[i]`.

---

## 3. Operator Laws

The laws an algorithm depends on are ordered by implication:

\f[
\text{spd} \Rightarrow \text{psd} \Rightarrow \text{self-adjoint}
\f]

```cpp
struct MyOperator {
    using laws = num::law::list<num::law::spd>;
    using domain_type = num::vec;
    using codomain_type = num::vec;

    num::idx rows() const;
    num::idx cols() const;
    void apply(const num::vec& x, num::vec& y) const;
};

static_assert(num::spd_operator<MyOperator>);
static_assert(num::self_adjoint_operator<MyOperator>); // implied
```

See @ref page_concepts for the laws, how to attach one, and what each routine requires.
