/// @file verlet_list.hpp
/// @brief Verlet neighbour list with skin-radius temporal caching
///
/// @par Algorithm (CS theory: amortized analysis via skin-radius invariant)
///
/// Building a cell list costs O(n + C) every timestep.  For large n this
/// is non-trivial.  The Verlet list amortizes this over multiple steps:
///
///   build()  -- O(n*k_ext) using the underlying CellList2D
///     Store, for each particle i, all neighbours j within the
///     *extended* cutoff r_ext = cutoff + skin.  Also snapshot positions.
///
///   needs_rebuild(pos)  -- O(n) per timestep
///     If max |Deltar_i| > skin/2 since last build -> return true.
///     Invariant: while max displacement < skin/2, every true neighbour
///     within cutoff is guaranteed to still be in the cached list.
///     (Any particle that has entered the cutoff from outside must have
///     crossed the skin shell first, which triggers a rebuild.)
///
///   neighbors(i)  -- O(1) per query, O(k) inner-loop iteration
///     Returns the cached IntRange for particle i.
///     Caller still checks |r_ij| < cutoff to skip stale far entries.
///
/// @par When does Verlet pay off?
///
///   Let tau = steps between rebuilds (determined by skin and particle speed).
///   Cost per step with Verlet: O(n*k_ext / tau + n*k_ext)
///                                      up build      up force eval (same work)
///   Cost per step without Verlet: O(n + C + n*k)
///                                         up cell list rebuild each step
///
///   Verlet wins when tau is large (slow physics, small dt, big skin) and n
///   is large enough that O(n+C) build cost is significant relative to
///   O(n*k) force eval.  For SPH with c0=10 m/s, dt=1 ms, h=0.025 m the
///   skin choice ~0.5h gives tau ~= 1-3 steps  -- borderline.  For softer
///   (slower) physics or larger n, the benefit is clear.
///
/// @tparam Scalar  float or double
#pragma once

#include "cell_list.hpp"
#include <vector>
#include <cmath>
#include <utility>

namespace num {

template<typename Scalar>
class VerletList2D {
public:
    /// @param cutoff      Interaction cutoff (e.g. 2h for SPH).
    /// @param skin        Skin thickness added to cutoff for the cached list.
    ///                    Larger skin -> fewer rebuilds but more cache entries.
    ///                    Typical: 0.3*cutoff (molecular dynamics) or
    ///                             0.5*cutoff (SPH, faster particle motion).
    VerletList2D(Scalar cutoff, Scalar skin)
        : cutoff_(cutoff), skin_(skin)
        , ext_sq_((cutoff + skin) * (cutoff + skin))
    {}

    /// @brief Build the neighbour list using a pre-built CellList2D.
    ///
    /// The cell list must have been built with cell_size >= cutoff + skin so
    /// that a single 3x3 query covers the full extended cutoff.
    ///
    /// PosAccessor: callable int -> std::pair<Scalar, Scalar>
    template<typename PosAccessor>
    void build(PosAccessor&& get_pos, int n, const CellList2D<Scalar>& cl) {
        starts_.resize(n + 1);
        flat_.clear();
        ref_x_.resize(n);
        ref_y_.resize(n);

        starts_[0] = 0;
        for (int i = 0; i < n; ++i) {
            auto [xi, yi] = get_pos(i);
            ref_x_[i] = xi;
            ref_y_[i] = yi;

            cl.query(xi, yi, [&](int j) {
                if (j == i) return;
                auto [xj, yj] = get_pos(j);
                const Scalar dx = xi - xj, dy = yi - yj;
                if (dx * dx + dy * dy < ext_sq_)
                    flat_.push_back(j);
            });

            starts_[i + 1] = static_cast<int>(flat_.size());
        }
    }

    /// @brief Check whether any particle has moved far enough to invalidate
    ///        the cached list.  O(n), very cheap (just arithmetic, no memory
    ///        allocation).
    ///
    /// Returns true if the cell list and Verlet list must be rebuilt before
    /// the next force evaluation.
    template<typename PosAccessor>
    bool needs_rebuild(PosAccessor&& get_pos, int n) const {
        if (ref_x_.empty()) return true;
        const Scalar half_skin_sq = (skin_ * Scalar(0.5)) * (skin_ * Scalar(0.5));
        for (int i = 0; i < n; ++i) {
            auto [xi, yi] = get_pos(i);
            const Scalar dx = xi - ref_x_[i];
            const Scalar dy = yi - ref_y_[i];
            if (dx * dx + dy * dy > half_skin_sq) return true;
        }
        return false;
    }

    /// @brief Cached neighbours of particle i (within cutoff + skin).
    ///
    /// Caller should still distance-filter to the true cutoff.
    IntRange neighbors(int i) const noexcept {
        return { flat_.data() + starts_[i],
                 flat_.data() + starts_[i + 1] };
    }

    Scalar cutoff()    const noexcept { return cutoff_; }
    Scalar skin()      const noexcept { return skin_; }
    Scalar ext_cutoff() const noexcept { return cutoff_ + skin_; }
    int    n_particles() const noexcept {
        return starts_.empty() ? 0 : static_cast<int>(starts_.size()) - 1;
    }

private:
    Scalar cutoff_, skin_, ext_sq_;
    std::vector<int>    flat_;    ///< Flat neighbour storage (all particles)
    std::vector<int>    starts_;  ///< starts_[i] = begin of particle i in flat_
    std::vector<Scalar> ref_x_;   ///< Positions at last build (for displacement check)
    std::vector<Scalar> ref_y_;
};

} // namespace num
