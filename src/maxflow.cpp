#include "gsd/maxflow.h"

#include <algorithm>
#include <limits>
#include <queue>

namespace gsd {

MaxFlow::MaxFlow(int n_nodes) : n_(n_nodes), out_(n_nodes), level_(n_nodes), next_(n_nodes) {}

void MaxFlow::add_edge(int u, int v, std::int64_t cap, std::int64_t reverse_cap) {
    out_[u].push_back(static_cast<int>(arcs_.size()));
    arcs_.push_back({v, cap});
    out_[v].push_back(static_cast<int>(arcs_.size()));
    arcs_.push_back({u, reverse_cap});
}

bool MaxFlow::bfs(int s, int t) {
    std::fill(level_.begin(), level_.end(), -1);
    std::queue<int> queue;
    level_[s] = 0;
    queue.push(s);
    while (!queue.empty()) {
        const int u = queue.front();
        queue.pop();
        for (int a : out_[u])
            if (arcs_[a].cap > 0 && level_[arcs_[a].to] < 0) {
                level_[arcs_[a].to] = level_[u] + 1;
                queue.push(arcs_[a].to);
            }
    }
    return level_[t] >= 0;
}

// Blocking flow by an iterative depth-first search (the graphs are too deep for recursion).
std::int64_t MaxFlow::dfs(int s, int t, std::int64_t limit) {
    std::int64_t total = 0;
    std::vector<int> path;  // arcs from s to the current node
    int u = s;
    while (true) {
        if (u == t) {
            std::int64_t push = limit - total;
            for (int a : path) push = std::min(push, arcs_[a].cap);
            for (int a : path) {
                arcs_[a].cap -= push;
                arcs_[a ^ 1].cap += push;
            }
            total += push;
            if (total == limit) return total;
            // Back to the first saturated arc of the path.
            std::size_t keep = 0;
            while (keep < path.size() && arcs_[path[keep]].cap > 0) keep++;
            path.resize(keep);
            u = path.empty() ? s : arcs_[path.back()].to;
            continue;
        }
        bool advanced = false;
        for (int& i = next_[u]; i < static_cast<int>(out_[u].size()); i++) {
            const int a = out_[u][i];
            if (arcs_[a].cap > 0 && level_[arcs_[a].to] == level_[u] + 1) {
                path.push_back(a);
                u = arcs_[a].to;
                advanced = true;
                break;
            }
        }
        if (advanced) continue;
        if (path.empty()) return total;
        // Dead end: this node is not used again in this phase.
        level_[u] = -1;
        const int a = path.back();
        path.pop_back();
        u = arcs_[a ^ 1].to;
    }
}

std::int64_t MaxFlow::run(int source, int sink) {
    std::int64_t flow = 0;
    while (bfs(source, sink)) {
        std::fill(next_.begin(), next_.end(), 0);
        flow += dfs(source, sink, std::numeric_limits<std::int64_t>::max());
    }
    return flow;
}

std::vector<char> MaxFlow::source_side(int source) const {
    std::vector<char> reachable(n_, 0);
    std::vector<int> stack = {source};
    reachable[source] = 1;
    while (!stack.empty()) {
        const int u = stack.back();
        stack.pop_back();
        for (int a : out_[u])
            if (arcs_[a].cap > 0 && !reachable[arcs_[a].to]) {
                reachable[arcs_[a].to] = 1;
                stack.push_back(arcs_[a].to);
            }
    }
    return reachable;
}

}  // namespace gsd
