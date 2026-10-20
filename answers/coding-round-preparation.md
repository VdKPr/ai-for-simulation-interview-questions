# Coding Round Preparation

[← Questions](../README.md) · [All answers](README.md)

**Short answer.** Expect two kinds of coding rounds. **General algorithm rounds** (PhysicsX candidates report three problems on Coderbyte, and a separate 30-minute data-structures round) test standard data structures: hash maps, sorting, heaps, stacks, graphs with BFS/DFS, union-find, binary search, two pointers and sliding windows, and simple dynamic programming. **Domain rounds** test the same skills on simulation data: meshes, point clouds, small solvers and ML building blocks (see [Mesh data processing](mesh-data-processing.md) and the "Code:" sections linked from the questions). Many algorithm problems in this field are graph problems in disguise, because a mesh is a graph. Practise talking while you code: clarify the input, state the complexity before writing, write a correct simple version first, then test it on edge cases out loud.

**On this page**

- [What to practise](#what-to-practise)
- [How to work through a problem](#how-to-work-through-a-problem)
- [Graph problems on meshes](#graph-problems-on-meshes)
- [Union-find for connected parts](#union-find-for-connected-parts)
- [Heaps, sorting and intervals](#heaps-sorting-and-intervals)
- [Binary search and sliding windows](#binary-search-and-sliding-windows)
- [Dynamic programming](#dynamic-programming)
- [Python details that cost time](#python-details-that-cost-time)
- [In an interview](#in-an-interview)

## What to practise

| Topic | Typical problems | Simulation flavour |
| --- | --- | --- |
| Hash maps and sets | Two-sum, duplicates, grouping, counting | Welding vertices, counting edges, deduplicating designs |
| Sorting | Merge intervals, k-th largest, custom keys | Sorting edges, ordering time steps |
| Heaps | Top-k, merge k sorted lists, Dijkstra | k hottest elements, shortest paths on a mesh |
| Graphs: BFS and DFS | Connected components, shortest path in hops, cycle detection, topological sort | k-ring neighbourhoods, separate parts in an STL, solver task dependencies |
| Union-find | Number of components, merging groups | Parts of a mesh, contact groups |
| Binary search | Search in sorted arrays, search on the answer | Look-up in tabulated material data, finding a time step |
| Two pointers, sliding window | Longest substring, max sum window | Moving averages of sensor signals, peak windows |
| Dynamic programming | Climbing stairs, coin change, edit distance, longest increasing subsequence | Rarely domain-specific; know the classics |
| Matrices | Rotate, spiral, flood fill, islands | Grid fields, voxel occupancy |

Platforms: [LeetCode](https://leetcode.com/), [NeetCode's roadmap](https://neetcode.io/roadmap) and [Coderbyte](https://coderbyte.com/) (the platform reported for PhysicsX). About 75–150 well-understood problems across these topics is enough for most rounds.

## How to work through a problem

1. **Restate** the problem and **clarify** inputs: sizes, types, empty input, duplicates, negative numbers, whether indices start at 0.
2. **Give a brute-force idea** and its complexity in one sentence.
3. **Improve it:** which data structure removes the repeated work? State the target complexity.
4. **Write clean code:** clear names, small helper functions, no premature tricks.
5. **Test out loud:** a normal case, an edge case (empty, one element), and a tricky case.
6. **State the final time and memory complexity**, and what you would change for very large inputs.

## Graph problems on meshes

A triangle mesh gives a graph whose nodes are vertices and whose edges are triangle edges. Two common questions: the **k-ring** (all vertices within k edges of a vertex, used for local features and smoothing) and the **shortest path in hops**. Both are breadth-first search: O(V + E).

```python
from collections import defaultdict, deque

def mesh_graph(faces):
    adj = defaultdict(set)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            adj[u].add(v); adj[v].add(u)
    return adj

def k_ring(adj, start, k):
    """All vertices within k edges of start, with their hop distance (breadth-first search)."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if dist[u] == k:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

# a strip of 8 triangles: vertices 0..9 in two rows (0 2 4 6 8 on top, 1 3 5 7 9 below)
faces = [(i, i + 1, i + 2) for i in range(8)]
adj = mesh_graph(faces)
print("1-ring of vertex 0:", sorted(k_ring(adj, 0, 1)), "| 2-ring:", sorted(k_ring(adj, 0, 2)))
print("hops from vertex 0 to vertex 9:", k_ring(adj, 0, 10)[9])
```

Output:

```
1-ring of vertex 0: [0, 1, 2] | 2-ring: [0, 1, 2, 3, 4]
hops from vertex 0 to vertex 9: 5
```

**Weighted shortest paths** (edge lengths instead of hops) use Dijkstra's algorithm with a heap: O(E log V). That gives an approximate geodesic distance on the surface ([Geodesic convolutions](geodesic-convolutions-and-shape-optimization.md)).

## Union-find for connected parts

An STL often contains several disconnected parts (a body and its wheels, a bracket and its bolts). **Union-find** (disjoint sets) finds them in nearly linear time: union the vertices of every face, then count roots. Path compression and union by size make each operation effectively constant time.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n)); self.size = [1] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]       # path halving
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra; self.size[ra] += self.size[rb]

def connected_parts(n_vertices, faces):
    uf = UnionFind(n_vertices)
    for a, b, c in faces:
        uf.union(a, b); uf.union(b, c)
    groups = defaultdict(list)
    for v in range(n_vertices):
        groups[uf.find(v)].append(v)
    return list(groups.values())

two_parts = [(0, 1, 2), (0, 2, 3), (4, 5, 6)]           # a quad (two triangles) and a separate triangle
print("parts:", connected_parts(7, two_parts))
```

Output:

```
parts: [[0, 1, 2, 3], [4, 5, 6]]
```

## Heaps, sorting and intervals

**Top-k** with a heap is O(n log k), better than sorting everything when k is small: for example, the k elements with the highest stress. **Merging intervals** (sort by start, then sweep) appears as merging time windows or overlapping ranges of design parameters.

```python
import heapq

def top_k(values, k):
    """Indices of the k largest values: O(n log k) with a min-heap of size k."""
    heap = []
    for i, v in enumerate(values):
        if len(heap) < k:
            heapq.heappush(heap, (v, i))
        elif v > heap[0][0]:
            heapq.heapreplace(heap, (v, i))
    return [i for v, i in sorted(heap, reverse=True)]

def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out

stress = [120.0, 310.5, 95.2, 410.0, 305.1, 88.0]
print("3 highest-stress elements:", top_k(stress, 3))
print("merged windows:", merge_intervals([(0, 2), (5, 7), (1, 3), (6, 9), (12, 13)]))
```

Output:

```
3 highest-stress elements: [3, 1, 4]
merged windows: [[0, 3], [5, 9], [12, 13]]
```

## Binary search and sliding windows

**Binary search** works on any monotone condition, not only sorted arrays: "the first time step where temperature exceeds the limit", or "the smallest mesh size that meets a tolerance". Python's `bisect` covers sorted look-ups; write the loop yourself for "search on the answer". **Sliding windows** keep a running sum or a deque of candidates, giving O(n) instead of O(n·w).

```python
from bisect import bisect_left

def interpolate_table(xs, ys, x):
    """Linear interpolation in a sorted table (e.g. a material property against temperature)."""
    i = min(max(bisect_left(xs, x), 1), len(xs) - 1)
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])

def first_true(lo, hi, condition):
    """Smallest integer in [lo, hi] where a monotone condition becomes True."""
    while lo < hi:
        mid = (lo + hi) // 2
        if condition(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def max_window_mean(signal, w):
    s = best = sum(signal[:w])
    for i in range(w, len(signal)):
        s += signal[i] - signal[i - w]
        best = max(best, s)
    return best / w

temps, conductivity = [20, 100, 200, 400], [50.0, 48.0, 45.0, 40.0]
print("conductivity at 150 C:", interpolate_table(temps, conductivity, 150))
print("first step with T > 300 (T = 20 + 7 * step):", first_true(0, 1000, lambda s: 20 + 7 * s > 300))
print("max 3-sample mean:", round(max_window_mean([1, 3, 2, 8, 7, 1, 0, 9], 3), 3))
```

Output:

```
conductivity at 150 C: 46.5
first step with T > 300 (T = 20 + 7 * step): 41
max 3-sample mean: 5.667
```

## Dynamic programming

DP problems appear less often in this field, but a coding platform round may include one. Identify the **state**, the **transition** and the **base case**; start with recursion plus memoization, then convert to a table if asked. Example: the minimum number of coins (or, equivalently, standard part lengths) that sum to a target.

```python
def min_pieces(lengths, target):
    """Fewest standard pieces summing exactly to target, or -1 (unbounded coin change)."""
    INF = float("inf")
    best = [0] + [INF] * target
    for t in range(1, target + 1):
        for L in lengths:
            if L <= t and best[t - L] + 1 < best[t]:
                best[t] = best[t - L] + 1
    return best[target] if best[target] < INF else -1

print("fewest pieces of lengths 1, 5, 7 summing to 18:", min_pieces([1, 5, 7], 18),
      "| lengths 4, 6 summing to 7:", min_pieces([4, 6], 7))
```

Output:

```
fewest pieces of lengths 1, 5, 7 summing to 18: 4 | lengths 4, 6 summing to 7: -1
```

## Python details that cost time

- `collections`: `defaultdict`, `Counter`, `deque` (O(1) pops from both ends; `list.pop(0)` is O(n)).
- `heapq` is a **min**-heap; push negatives or (key, item) tuples for a max-heap.
- Sorting with keys: `sorted(items, key=lambda x: (x[1], -x[0]))`.
- Recursion depth is limited (about 1,000 by default); use an explicit stack for deep DFS on large meshes.
- Do not modify a list while iterating over it; do not use a mutable default argument (`def f(x=[])`).
- Integer division `//` and modulo with negatives behave differently from C.
- Online platforms often read from standard input: practise `sys.stdin` parsing and printing exactly the required format.
- For NumPy rounds: no Python loops over elements; know `np.unique(..., return_inverse=True)`, `np.add.at`, `np.argsort`, broadcasting and fancy indexing ([Making Python fast](mesh-data-processing.md#making-python-fast-on-millions-of-points)).

## In an interview

**Thirty-second answer to "How did you prepare for coding?"** "I practised the standard topics, hash maps, heaps, graphs with BFS and union-find, binary search and a little DP, and then the same ideas on simulation data: mesh adjacency, connected parts of an STL, k-rings, top-k elements. In the round I clarify the input, state a brute-force solution and its complexity, improve it, write it cleanly and test edge cases out loud."

**Likely follow-ups**

- *What is the complexity, and what happens with 10 million vertices?* BFS and union-find are linear; switch to NumPy/SciPy (`scipy.sparse.csgraph.connected_components`) for speed and memory.
- *How would you test this?* Small hand-made cases with known answers, edge cases (empty, a single element, disconnected input), and a comparison with a brute-force version on random input.

## References

- [NeetCode roadmap](https://neetcode.io/roadmap) and [LeetCode](https://leetcode.com/)
- T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed., MIT Press, 2022.
- [Python `heapq`](https://docs.python.org/3/library/heapq.html), [`bisect`](https://docs.python.org/3/library/bisect.html) and [`collections`](https://docs.python.org/3/library/collections.html)
- [SciPy `csgraph`](https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html)
- Interview reports: [PhysicsX on Glassdoor](https://www.glassdoor.com.mx/Entrevista/PhysicsX-Entrevista-E9306867-RVW97800229.htm) and [Industry notes](../industry.md#reported-interview-formats)
