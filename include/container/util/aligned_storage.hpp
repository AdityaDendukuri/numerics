/// @file container/util/aligned_storage.hpp
/// @brief Over-aligned owning storage for the dense containers.
///
/// The guarantee covers the base pointer only.
#pragma once

#include "core/types.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

/// @brief Storage alignment in bytes, 64 by default: a cache line on x86-64 and a multiple of
/// every common SIMD width. Override with `-DNUMERICS_STORAGE_ALIGNMENT=<n>`.
#ifndef NUMERICS_STORAGE_ALIGNMENT
#define NUMERICS_STORAGE_ALIGNMENT 64
#endif

namespace num {

/// @brief Alignment, in bytes, of the storage owned by every dense container. It enters the
/// deallocation call, so it must match across translation units; the CMake target sets it
/// `PUBLIC` for that reason.
inline constexpr std::size_t storage_alignment = NUMERICS_STORAGE_ALIGNMENT;

static_assert(storage_alignment >= alignof(std::max_align_t),
              "NUMERICS_STORAGE_ALIGNMENT must be at least the default new alignment");
static_assert((storage_alignment & (storage_alignment - 1)) == 0,
              "NUMERICS_STORAGE_ALIGNMENT must be a power of two");

namespace detail {

/// @brief Releases storage obtained from `allocate_aligned`. It carries the count to run the
/// destructors and call the matching aligned `operator delete`.
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
        // the unsized aligned form: clang before 19 needs -fsized-deallocation for the sized one
        ::operator delete(static_cast<void *>(pointer), std::align_val_t{storage_alignment});
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
    (void)count; // see aligned_deleter
    ::operator delete(static_cast<void *>(pointer), std::align_val_t{storage_alignment});
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

/// @brief Allocate `count` default-initialized elements on an aligned boundary, like
/// `new T[count]`. Scalar contents are indeterminate, so use it only when the caller overwrites
/// the whole buffer.
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

/// @brief Restate a container's storage alignment for the optimizer. Valid only on the base
/// pointer of storage obtained above; null passes through.
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
