// Maximum flow with integer capacities (Dinic), and the source side of the minimum cut.
#pragma once
#include <cstdint>
#include <vector>

namespace gsd {

class MaxFlow {
public:
    explicit MaxFlow(int n_nodes);
    // Edge u -> v with capacity cap, and v -> u with capacity reverse_cap.
    void add_edge(int u, int v, std::int64_t cap, std::int64_t reverse_cap = 0);
    std::int64_t run(int source, int sink);
    // After run: the nodes reachable from the source in the residual graph. It is the smallest
    // source side among the minimum cuts, the same for every maximum flow.
    std::vector<char> source_side(int source) const;

private:
    struct Arc { int to; std::int64_t cap; };
    int n_;
    std::vector<Arc> arcs_;                 // arc i and its reverse i ^ 1
    std::vector<std::vector<int>> out_;
    std::vector<int> level_, next_;
    bool bfs(int s, int t);
    std::int64_t dfs(int u, int t, std::int64_t pushed);
};

}  // namespace gsd
