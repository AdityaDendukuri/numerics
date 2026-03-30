/// @file spatial/connected_components.hpp
/// @brief Iterative BFS connected-component labelling.
///
/// connected_components(n_sites, in_cluster, neighbors)
///
/// Template parameters:
///   InCluster  callable: bool(int i)        -- true if site i is in the cluster
///   Neighbors  callable: void(int i, F&& f) -- calls f(nb) for each neighbor nb of i
///
/// Returns ClusterResult where id[i] is:
///   -2  = excluded (in_cluster returned false)
///   >=0 = cluster index
///
/// ClusterResult::largest_id and largest_size track the biggest connected component.
#pragma once

#include <vector>

namespace num {

struct ClusterResult {
    std::vector<int> id;        ///< Per-site label: -2 excluded, >=0 cluster index
    std::vector<int> sizes;     ///< sizes[c] = number of sites in cluster c
    int largest_id   = -1;      ///< Index of largest cluster (-1 if none)
    int largest_size = 0;       ///< Size of largest cluster
};

/// BFS connected-component labelling with pre-allocated flat queue (no heap per call).
///
/// @param n_sites   Total number of sites
/// @param in_cluster  bool(int i) -- include site i?
/// @param neighbors   void(int i, auto&& visit) -- call visit(nb) per neighbor of i
template<typename InCluster, typename Neighbors>
ClusterResult connected_components(int n_sites,
                                    InCluster&&  in_cluster,
                                    Neighbors&&  neighbors) {
    ClusterResult res;
    res.id.resize(n_sites);
    res.sizes.reserve(64);

    for (int i = 0; i < n_sites; ++i)
        res.id[i] = in_cluster(i) ? -1 : -2;   // -1 = unvisited included, -2 = excluded

    std::vector<int> queue(n_sites);
    int qhead = 0, qtail = 0;

    for (int start = 0; start < n_sites; ++start) {
        if (res.id[start] != -1) continue;

        const int cid = static_cast<int>(res.sizes.size());
        res.sizes.push_back(0);
        res.id[start]  = cid;
        queue[qtail++] = start;

        while (qhead < qtail) {
            const int i = queue[qhead++];
            ++res.sizes[cid];
            neighbors(i, [&](int nb) {
                if (res.id[nb] == -1) { res.id[nb] = cid; queue[qtail++] = nb; }
            });
        }

        if (res.sizes[cid] > res.largest_size) {
            res.largest_size = res.sizes[cid];
            res.largest_id   = cid;
        }
    }
    return res;
}

} // namespace num
