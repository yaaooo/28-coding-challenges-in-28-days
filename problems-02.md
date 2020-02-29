## Day 08: Partitioning an Array into K Equal Sum Subsets

From [LeetCode](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/):

> Given an array of integers nums and a positive integer k, find whether it's possible to divide this array into k non-empty subsets whose sums are all equal.
>
> ```py
> Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
> Output: True
> # Example: (5), (1, 4), (2,3), (2,3) have equal sums
> ```
> Note:
> - 1 <= k <= len(nums) <= 16.
> - 0 < nums[i] < 10000.
>

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/partitioning-an-array-into-k-equal-sum-subsets-python/):

```py
def can_partition(nums, k):
    def has_ith_bit(x, i):
        return x & (1 << i)
        
    def set_ith_bit(x, i):
        return x | (1 << i)

    subset_tgt, remainder = divmod(sum(nums), k)
    if remainder != 0:
        return False

    num_subsets = 1 << len(nums)
    DP = [False for _ in range(num_subsets)]
    DP[0] = True
    
    subset_sums = [None for _ in range(num_subsets)]
    subset_sums[0] = 0

    for x in range(num_subsets):
        if DP[x]:
            for i, num in enumerate(nums):
                if not has_ith_bit(x, i):
                    x_with_num = set_ith_bit(x, i)
                    subset_sums[x_with_num] = subset_sums[x] + num
                    DP[x_with_num] = (
                        (subset_sums[x] % subset_tgt) + num <= subset_tgt
                    )

    return DP[-1]
```

</details>

## Day 09: Maximum Profit from K Transactions

From [LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/):

> You have an array for which the i-th element is the price of a given stock on day i. Design an algorithm to find the maximum profit. You may complete __at most__ k transactions.
>
> Note: You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
> 
> ```py
> Input: prices = [3, 2, 6, 5, 0, 3], k = 2
> Output: 7
> # Explanation:
> # 1. Buy on day 2 and sell on day 3, profit = 6 - 2 = 4.
> # 2. Buy on day 5 and sell on day 6, profit = 3 - 0 = 3.
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/maximum-profit-from-k-transactions-python/):

```py
def max_profit(k, prices):
    if not k or not prices:
        return 0

    n = len(prices)

    # (Optional) Additional optimisation if we
    # are able to execute unlimited transactions
    if k >= len(prices) * 2:
        max_profits = 0
        for i in range(1, n):
            max_profits += max(prices[i] - prices[i-1], 0)
        return max_profits


    # Core DP algorithm
    profits = [0 for _ in range(n)]

    for txn_idx in range(1, k + 1):
        new_profits = profits[:]
        best_preceding_value = -prices[0]

        for sell_idx in range(1, n):
            new_profits[sell_idx] = max(
                new_profits[sell_idx-1],
                best_preceding_value + prices[sell_idx]
            )

            best_preceding_value = max(
                best_preceding_value,
                profits[sell_idx-1] - prices[sell_idx]
            )

        profits = new_profits

    return profits[-1]
```

</details>

## Day 10: Connecting Cities at Minimum constant

From [LeetCode](https://leetcode.com/problems/connecting-cities-with-minimum-cost/):

> There are `N` cities numbered from `1` to `N`.
>
> You are given array of `connections`. Each connection, `[c1, c2, cost]`, describes the `cost` of connecting `city1` and `city2` together. A connection is  bidirectional –– connecting `city1` and `city2` is the same as connecting `city2` and `city1`.
>
> Return the minimum cost such that for every pair of cities, there exists a path of connections (possibly of length `1`) that connects them together.
> The cost is the sum of the connection costs used. If the task is impossible, return `-1`.
>
> ```py
> Input: N = 3, connections = [[1, 2, 5], [1, 3, 6], [2, 3, 1]]
> Output: 6
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/connecting-cities-at-minimum-cost-python/):

```py
import heapq
import collections

class Node:
    def __init__(self, label):
        self.label = label
        self.edges = set()


Edge = collections.namedtuple('Edge', ('cost', 'label'))


def min_MST_cost(n, connections):
    nodes = {i: Node(i) for i in range(1, n+1)}
    for c1, c2, cost in connections:
        nodes[c1].edges.add(Edge(cost, c2))
        nodes[c2].edges.add(Edge(cost, c1))

    pq = [Edge(0, 1)]
    visited = set()
    total_cost = 0

    while pq and len(visited) < n:
        popped = heapq.heappop(pq)

        if popped.label not in visited:
            visited.add(popped.label)
            node = nodes[popped.label]
            total_cost += popped.cost

            for edge in node.edges:
                heapq.heappush(pq, edge)

    if len(visited) != n:
        return -1
    return total_cost
```

</details>

## Day 11: Largest Rectangle In Histogram

From [LeetCode](https://leetcode.com/problems/largest-rectangle-in-histogram/):

> Given an array of non-negative integers which represent histogram bar heights, find the area of largest rectangle.
>
> ```py
> Input: [2, 1, 5, 6, 2, 3]
> """
>       #
>     # #
>     # #
>     # #   #
> #   # # # #
> # # # # # #
> 0 1 2 3 4 5
> """
> Output: 10
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/largest-rectangle-in-histogram-python/):

```py
def get_largest_rectangle_area(heights):

    active = []
    largest_area = 0

    for i, new_height in enumerate(heights + [0]):
        while active and heights[active[-1]] >= new_height:
            popped_idx = active.pop()

            height = heights[popped_idx]

            leftbound = active[-1] if active else -1
            rightbound = i   
            width = rightbound - leftbound - 1

            largest_area = max(largest_area, width * height)
            
        active.append(i)

    return largest_area
```

</details>

## Day 12: Trapping Rain Water

From [LeetCode](https://leetcode.com/problems/trapping-rain-water/):

> Given an array of non-negative integers that represent an elevation map where the width of each bar is 1, compute the amount of rainwater that can be trapped within it.
>
> ```py
> Input: [0, 1, 0, 3, 1, 0, 1, 2]
> """
>       #
>       # ~ ~ ~ #
>   # ~ # # ~ # #
> 0 1 2 3 4 5 6 7
> """
> Output: 5
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/trapping-rain-water-python/):

```py
def trap(heights):

    max_left, max_right = 0, 0
    left, right = 0, len(heights) - 1 
    
    vol = 0
    while left < right:
        max_left = max(max_left, heights[left])
        max_right = max(max_right, heights[right])
        
        if max_left <= max_right:
            vol += max_left - heights[left] 
            left += 1
        
        else: # max_right < max_left
            vol += max_right - heights[right]
            right -= 1
            
    
    return vol
```

</details>

## Day 13: Trapping Rain Water (2D)

From [LeetCode](https://leetcode.com/problems/trapping-rain-water-ii/):

> Given an `n` x `m` matrix of positive integers representing the height of each cell in a 2D elevation map, compute the amount of water that can be trapped within it.
>
> Note: Both `n` and `m` are less than 110. The height of each cell is greater than 0 and is less than 20,000.
> ```py
> Input:
> [
>  [1, 4, 3, 1, 3, 2],
>  [3, 2, 1, 3, 2, 4],
>  [2, 3, 3, 2, 3, 1]
> ]
> Output: 4
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/trapping-rain-water-2D-python/):

```py
import heapq
import collections 

Cell = collections.namedtuple('Cell', ('ht', 'r', 'c'))

SHIFT = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def trap(mtx):
    if not mtx:
        return 0
    
    n, m = len(mtx), len(mtx[0])

    # Store perimeter cells in min heap
    pq = []
    seen = set()
    for r in range(n):
        if r in [0, n-1]:
            for c in range(m):
                heapq.heappush(pq, Cell(mtx[r][c], r, c))
                seen.add((r, c))
        else:
            heapq.heappush(pq, Cell(mtx[r][0], r, 0))
            seen.add((r, 0))

            heapq.heappush(pq, Cell(mtx[r][m-1], r, m-1))
            seen.add((r, m-1))

    # Go through every seen cell stored in the heap
    water = 0
    popped_max = 0 # Upper bound value
    while pq:
        # Greedily pop off the lowest ones
        popped = heapq.heappop(pq)
        popped_max = max(popped_max, popped.ht)
        
        # Examine adjacent cells
        for dr, dc in SHIFT:
            nr, nc = popped.r + dr, popped.c + dc
            if (
                0 <= nr <= n - 1 and
                0 <= nc <= m - 1 and
                (nr, nc) not in seen
            ):
                # Increment water count if upper bound value
                # is greater than the adjacent cell's height
                water += max(popped_max - mtx[nr][nc], 0)
                heapq.heappush(pq, Cell(mtx[nr][nc], nr, nc))
                seen.add((nr, nc))

    return water
```

</details>

## Day 14: Finding Bridges in a Graph

From [LeetCode](https://leetcode.com/problems/critical-connections-in-a-network/):

> There are n servers in a network, connected by undirected server-to-server connections. A critical connection is a connection that, if removed, will make some server unable to reach some other server. Find all critical connections.
>
> ```py
> Input: n = 4, connections = [[0, 1], [1, 2], [2, 0], [1, 3]]
> Output: [[1, 3]] # [[3,1]] is also accepted.
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/finding-bridges-in-a-graph-python/):

```py
def find_bridges(n, connection):
    
    graph = collections.defaultdict(list)
    depths = [None for _ in range(n)]

    for a, b in connections:
        graph[a].append(b)
        graph[b].append(a)

    edges = set([
        tuple(sorted(c)) for c in connections
    ])

    def DFS(u, depth):
        
        if depths[u] is not None:
            return depths[u]

        depths[u] = depth

        min_cyclic_depth = math.inf

        for v in graph[u]:
            
            if depths[v] == depth-1:
                continue
                
            cyclic_depth = DFS(v, depth+1)
            if cyclic_depth <= depth:
                edges.discard(tuple(sorted([u, v])))
                min_cyclic_depth = min(
                    min_cyclic_depth, cyclic_depth
                )

        return min_cyclic_depth


    DFS(0, 0)
    return edges
```

</details>

