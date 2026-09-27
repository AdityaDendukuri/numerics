/// @file spatial/connected_components.hpp
/// @brief Iterative BFS connected-component labelling.
#pragma once

#include "core/types.hpp"
#include <ostream>
#include <vector>

namespace num {

struct cluster_result {
    array<int> id;    ///< Per-site label: -2 excluded, >=0 cluster index
    array<int> sizes; ///< sizes[c] = number of sites in cluster c
    int largest_id = -1;    ///< Index of largest cluster (-1 if none)
    int largest_size = 0;   ///< Size of largest cluster

    friend std::ostream &operator<<(std::ostream &os, const cluster_result &r) {
        os << "cluster_result{ num_clusters: " << r.sizes.size()
           << ", largest_id: " << r.largest_id
           << ", largest_size: " << r.largest_size << " }";
        return os;
    }
};

/// @brief Label connected components by BFS over one flat queue.
///
/// @param n_sites    Total number of sites
/// @param in_cluster bool(int i): include site i?
/// @param neighbors  void(int i, auto&& visit): call visit(nb) per neighbor of i
/// @return id[i] is -2 for an excluded site and the cluster index otherwise; also the
///         cluster sizes and the largest cluster.
template <typename InCluster, typename Neighbors>
cluster_result connected_components(int n_sites, InCluster &&in_cluster, Neighbors &&neighbors) {
    cluster_result res;
    res.id.resize(n_sites);
    res.sizes.reserve(64);

    for (int i = 0; i < n_sites; ++i) {
        res.id[i] = in_cluster(i) ? -1 : -2; // -1 = unvisited included, -2 = excluded
    }

    array<int> queue(n_sites);
    int qhead = 0, qtail = 0;

    for (int start = 0; start < n_sites; ++start) {
        if (res.id[start] != -1) {
            continue;
        }

        const int cid = static_cast<int>(res.sizes.size());
        res.sizes.push_back(0);
        res.id[start] = cid;
        queue[qtail++] = start;

        while (qhead < qtail) {
            const int i = queue[qhead++];
            ++res.sizes[cid];
            neighbors(i, [&](int nb) {
                if (res.id[nb] == -1) {
                    res.id[nb] = cid;
                    queue[qtail++] = nb;
                }
            });
        }

        if (res.sizes[cid] > res.largest_size) {
            res.largest_size = res.sizes[cid];
            res.largest_id = cid;
        }
    }
    return res;
}

} // namespace num
