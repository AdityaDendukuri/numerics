/// @file structures/graph/structured_grid.hpp
/// @brief Implicit weighted graph for a uniform Cartesian grid.
#pragma once

#include "core/types.hpp"
#include "structures/concepts.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace num {

/// Boundary policy for one axis of a structured grid graph.
enum class grid_boundary { reflecting, periodic };

/// An allocation-free graph view of a uniform D-dimensional Cartesian grid.
///
/// Vertices are cells, with axis 0 varying fastest. Edge weights are face
/// transmissibilities A/h, so a finite-volume diffusion jump rate is
/// `diffusivity * edge.weight / source_volume`. Reflecting boundaries omit the
/// exterior edge; periodic boundaries wrap to the opposite cell.
template <std::size_t D, typename Weight = double, std::integral Index = num::idx>
class structured_grid_graph {
    static_assert(D > 0, "structured_grid_graph needs at least one dimension");

  public:
    using weight_type = Weight;
    using index_type = Index;
    using coordinate_type = std::array<Index, D>;

    struct graph_edge {
        Index to{};
        Weight weight{1};
    };

    class neighbor_range {
      public:
        using const_iterator = typename std::array<graph_edge, 2 * D>::const_iterator;

        [[nodiscard]] const_iterator begin() const noexcept { return edges_.begin(); }
        [[nodiscard]] const_iterator end() const noexcept {
            return edges_.begin() + static_cast<std::ptrdiff_t>(size_);
        }
        [[nodiscard]] std::size_t size() const noexcept { return size_; }

      private:
        friend class structured_grid_graph;
        std::array<graph_edge, 2 * D> edges_{};
        std::size_t size_ = 0;
    };

    explicit structured_grid_graph(
        coordinate_type dimensions, std::array<Weight, D> spacing = unit_spacing(),
        std::array<grid_boundary, D> boundaries = reflecting_boundaries())
        : dimensions_(dimensions), spacing_(spacing), boundaries_(boundaries) {
        Index product = 1;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (dimensions_[axis] <= 0) {
                throw std::invalid_argument("structured grid dimensions must be positive");
            }
            if (!(spacing_[axis] > Weight{0}) || !std::isfinite(spacing_[axis])) {
                throw std::invalid_argument("structured grid spacing must be finite and positive");
            }
            strides_[axis] = product;
            if (dimensions_[axis] > std::numeric_limits<Index>::max() / product) {
                throw std::overflow_error("structured grid vertex count overflows its index type");
            }
            product *= dimensions_[axis];
        }
        n_vertices_ = product;

        cell_volume_ = Weight{1};
        for (Weight h : spacing_)
            cell_volume_ *= h;
        for (std::size_t axis = 0; axis < D; ++axis) {
            transmissibility_[axis] = cell_volume_ / (spacing_[axis] * spacing_[axis]);
        }

        n_edges_ = 0;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (dimensions_[axis] == 1)
                continue;
            const Index lines = n_vertices_ / dimensions_[axis];
            const Index edges_per_line = boundaries_[axis] == grid_boundary::periodic
                                             ? dimensions_[axis]
                                             : dimensions_[axis] - 1;
            if (edges_per_line > 0 && lines > std::numeric_limits<Index>::max() / edges_per_line) {
                throw std::overflow_error("structured grid edge count overflows its index type");
            }
            const Index axis_edges = lines * edges_per_line;
            if (axis_edges > std::numeric_limits<Index>::max() - n_edges_) {
                throw std::overflow_error("structured grid edge count overflows its index type");
            }
            n_edges_ += axis_edges;
        }
    }

    [[nodiscard]] static constexpr std::array<Weight, D> unit_spacing() {
        std::array<Weight, D> result{};
        result.fill(Weight{1});
        return result;
    }

    [[nodiscard]] static constexpr std::array<grid_boundary, D> reflecting_boundaries() {
        std::array<grid_boundary, D> result{};
        result.fill(grid_boundary::reflecting);
        return result;
    }

    [[nodiscard]] Index n_vertices() const noexcept { return n_vertices_; }
    [[nodiscard]] Index n_edges() const noexcept { return n_edges_; }
    [[nodiscard]] bool is_directed() const noexcept { return false; }
    [[nodiscard]] const coordinate_type &dimensions() const noexcept { return dimensions_; }
    [[nodiscard]] const std::array<Weight, D> &spacing() const noexcept { return spacing_; }
    [[nodiscard]] const std::array<grid_boundary, D> &boundaries() const noexcept {
        return boundaries_;
    }
    [[nodiscard]] Weight cell_volume() const noexcept { return cell_volume_; }

    [[nodiscard]] coordinate_type coordinate(Index vertex) const {
        check_vertex(vertex);
        coordinate_type result{};
        for (std::size_t axis = D; axis-- > 0;) {
            result[axis] = vertex / strides_[axis];
            vertex %= strides_[axis];
        }
        return result;
    }

    [[nodiscard]] Index vertex(const coordinate_type &coordinate) const {
        Index result = 0;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (coordinate[axis] < 0 || coordinate[axis] >= dimensions_[axis]) {
                throw std::out_of_range("structured grid coordinate is outside the grid");
            }
            result += coordinate[axis] * strides_[axis];
        }
        return result;
    }

    [[nodiscard]] Index degree(Index vertex_index) const {
        return static_cast<Index>(neighbors(vertex_index).size());
    }

    [[nodiscard]] Weight weighted_degree(Index vertex_index) const {
        Weight result{};
        for (const auto &edge : neighbors(vertex_index))
            result += edge.weight;
        return result;
    }

    [[nodiscard]] neighbor_range neighbors(Index vertex_index) const {
        const auto cell = coordinate(vertex_index);
        neighbor_range result;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (dimensions_[axis] == 1)
                continue;

            if (cell[axis] > 0) {
                result.edges_[result.size_++] = {vertex_index - strides_[axis],
                                                 transmissibility_[axis]};
            } else if (boundaries_[axis] == grid_boundary::periodic) {
                result.edges_[result.size_++] = {vertex_index +
                                                     (dimensions_[axis] - 1) * strides_[axis],
                                                 transmissibility_[axis]};
            }

            if (cell[axis] + 1 < dimensions_[axis]) {
                result.edges_[result.size_++] = {vertex_index + strides_[axis],
                                                 transmissibility_[axis]};
            } else if (boundaries_[axis] == grid_boundary::periodic) {
                result.edges_[result.size_++] = {vertex_index -
                                                     (dimensions_[axis] - 1) * strides_[axis],
                                                 transmissibility_[axis]};
            }
        }
        return result;
    }

  private:
    void check_vertex(Index vertex) const {
        if (vertex < 0 || vertex >= n_vertices_) {
            throw std::out_of_range("structured grid vertex is outside the graph");
        }
    }

    coordinate_type dimensions_{};
    coordinate_type strides_{};
    std::array<Weight, D> spacing_{};
    std::array<Weight, D> transmissibility_{};
    std::array<grid_boundary, D> boundaries_{};
    Index n_vertices_ = 0;
    Index n_edges_ = 0;
    Weight cell_volume_{};
};

static_assert(concepts::incidence_structure<structured_grid_graph<1>>);
static_assert(concepts::incidence_structure<structured_grid_graph<2>>);
static_assert(concepts::incidence_structure<structured_grid_graph<3>>);

} // namespace num
