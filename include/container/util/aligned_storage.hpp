/// @file container/util/aligned_storage.hpp
/// @brief Over-aligned owning storage for the dense containers.
///
/// `new T[n]` guarantees only `__STDCPP_DEFAULT_NEW_ALIGNMENT__` — 16 bytes on
/// every platform this library targets. The dense containers instead allocate
/// through `make_aligned` below, which uses the aligned `operator new` (C++17)
/// to place the first element on a `storage_alignment` boundary.
///
/// ### What this is and is not worth
///
/// Measured on AArch64/NEON and cross-compiled to x86-64/AVX-512, clang emits
/// *identical* code for the level-1 kernels with and without the guarantee: it
/// already prefers the unaligned move forms, which carry no penalty on an
/// aligned address on any current microarchitecture. Do not expect the
/// `std::assume_aligned` in `data()` to speed up a loop today; it is there so
/// the annotation is correct and in place for the kernels that will need it.
///
/// What the alignment does buy is structural:
///   - a vector load never straddles a cache line, so no access is split across
///     two L1 lookups;
///   - two containers never share a line at their boundaries, which is what
///     makes a threaded reduction over adjacent buffers free of false sharing;
///   - it is a hard *precondition* for aligned SIMD intrinsics
///     (`_mm512_load_pd`) and non-temporal stores (`_mm512_stream_pd`), and for
///     efficient pinned-host-memory registration on the device paths.
///
/// It also costs something: the aligned `operator new` bypasses the small-size
/// fast path in at least libc++'s allocator, which on macOS measured roughly
/// 4x the latency of plain `new` for buffers under a kilobyte. That is only
/// visible to code that allocates inside a loop — which is a defect in that
/// code, not in this header.
///
/// ### Scope of the guarantee
///
/// It covers the *base* pointer. An interior pointer such as `A.data() + j` is
/// aligned only when `j * sizeof(T)` is a multiple of `storage_alignment`, and
/// nothing here claims otherwise; the assumption is attached to the base, and
/// the compiler reasons about the offset itself. A matrix row therefore starts
/// on a boundary only when its stride happens to; padding the stride would
/// change what `data()` means and is deliberately not done here.
#pragma once

#include "core/types.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

/// @brief Storage alignment in bytes. Override at configure time with
/// `-DNUMERICS_STORAGE_ALIGNMENT=<n>`.
///
/// 64 is the cache-line size on x86-64 and is a multiple of every SIMD register
/// width in common use (16 for NEON and SSE, 32 for AVX2, 64 for AVX-512), so
/// one value serves both purposes.
#ifndef NUMERICS_STORAGE_ALIGNMENT
#define NUMERICS_STORAGE_ALIGNMENT 64
#endif

namespace num {

/// @brief Alignment, in bytes, of the storage owned by every dense container.
///
/// This value participates in the deallocation call, so it must be identical in
/// every translation unit of a build. The CMake package sets it on the `numerics`
/// target as a `PUBLIC` definition for exactly that reason; headers copied out of
/// the tree fall back to the default above.
inline constexpr std::size_t storage_alignment = NUMERICS_STORAGE_ALIGNMENT;

static_assert(storage_alignment >= alignof(std::max_align_t),
              "NUMERICS_STORAGE_ALIGNMENT must be at least the default new alignment");
static_assert((storage_alignment & (storage_alignment - 1)) == 0,
              "NUMERICS_STORAGE_ALIGNMENT must be a power of two");

namespace detail {

/// @brief Releases storage obtained from `allocate_aligned`.
///
/// Carries the element count for two reasons: over-aligned storage must be
/// returned through the matching aligned `operator delete`, and a non-trivially
/// destructible element type needs its destructors run explicitly. The array
/// form of `new` would track both, but it cannot be given an alignment for an
/// element type whose own `alignof` is weaker.
template <class T>
class aligned_deleter {
  public:
    constexpr aligned_deleter() noexcept = default;
    constexpr explicit aligned_deleter(idx count) noexcept : count_(count) {}

    void operator()(T *pointer) const noexcept {
        if (pointer == nullptr) {
            return;
        }
        std::destroy_n(pointer, count_);
        ::operator delete(static_cast<void *>(pointer), count_ * sizeof(T),
                          std::align_val_t{storage_alignment});
    }

  private:
    idx count_ = 0;
};

/// @brief Obtain raw `storage_alignment`-aligned storage for `count` elements.
/// @throws std::bad_alloc If the byte count overflows `idx`, or on allocation failure.
template <class T>
[[nodiscard]] inline T *allocate_aligned(idx count) {
    static_assert(storage_alignment >= alignof(T),
                  "NUMERICS_STORAGE_ALIGNMENT is weaker than this element type requires");
    if (count > std::numeric_limits<idx>::max() / sizeof(T)) {
        throw std::bad_alloc();
    }
    return static_cast<T *>(::operator new(count * sizeof(T), std::align_val_t{storage_alignment}));
}

/// @brief Release raw storage whose elements were never constructed.
template <class T>
inline void deallocate_aligned(T *pointer, idx count) noexcept {
    ::operator delete(static_cast<void *>(pointer), count * sizeof(T),
                      std::align_val_t{storage_alignment});
}

} // namespace detail

/// @brief Owning handle to `storage_alignment`-aligned storage for `T`.
template <class T>
using aligned_array = std::unique_ptr<T[], detail::aligned_deleter<T>>;

/// @brief Allocate `count` value-initialized elements on an aligned boundary.
///
/// Equivalent to `new T[count]()`, except for the alignment and the overflow check.
/// @throws std::bad_alloc On overflow or allocation failure.
template <class T>
[[nodiscard]] inline aligned_array<T> make_aligned(idx count) {
    if (count == 0) {
        return aligned_array<T>(nullptr, detail::aligned_deleter<T>(0));
    }
    T *storage = detail::allocate_aligned<T>(count);
    try {
        std::uninitialized_value_construct_n(storage, count);
    } catch (...) {
        detail::deallocate_aligned(storage, count);
        throw;
    }
    return aligned_array<T>(storage, detail::aligned_deleter<T>(count));
}

/// @brief Allocate `count` default-initialized elements on an aligned boundary.
///
/// Equivalent to `new T[count]`: for a scalar element type the contents are
/// indeterminate, so every element must be written before it is read. Use this
/// where the caller immediately fills or copies over the whole buffer, to avoid
/// paying for a zero fill that is about to be overwritten.
/// @throws std::bad_alloc On overflow or allocation failure.
template <class T>
[[nodiscard]] inline aligned_array<T> make_aligned_for_overwrite(idx count) {
    if (count == 0) {
        return aligned_array<T>(nullptr, detail::aligned_deleter<T>(0));
    }
    T *storage = detail::allocate_aligned<T>(count);
    try {
        std::uninitialized_default_construct_n(storage, count);
    } catch (...) {
        detail::deallocate_aligned(storage, count);
        throw;
    }
    return aligned_array<T>(storage, detail::aligned_deleter<T>(count));
}

/// @brief Restate a container's storage alignment for the optimizer.
///
/// Valid only on the base pointer of storage obtained above. Null is passed
/// through untouched: `std::assume_aligned` requires a pointer to an object, and
/// an empty container owns none.
template <class T>
[[nodiscard]] inline T *assume_storage_aligned(T *pointer) noexcept {
    if (pointer == nullptr) {
        return nullptr;
    }
    return std::assume_aligned<storage_alignment>(pointer);
}

/// @brief True when `pointer` sits on a `storage_alignment` boundary.
///
/// Diagnostic only; the containers guarantee this by construction.
template <class T>
[[nodiscard]] inline bool is_storage_aligned(const T *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) % storage_alignment) == 0;
}

} // namespace num
