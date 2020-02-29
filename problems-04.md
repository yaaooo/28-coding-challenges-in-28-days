
## Day 22: Longest Subarray with a Sum Constraint

From [EPI](https://github.com/adnanaziz/EPIJudge/blob/master/epi_judge_python/longest_subarray_with_sum_constraint.py):

> Given an array of numbers `A` and a key `k`, find the length of the longest subarray whose sum is less than or equal to `k`.
>
> ```py
> Input: 
> A = [431, -15, 639, 342, -14, 565, -924, 635, 167, -70]
> k = 184
> 
> Output: 4 # sum(A[3:7]) <= 184
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/longest-subarray-with-a-sum-constraint-python/):

```py
def find_longest_subarray_less_equal_k(A, k):

    # Get prefix sums
    prefix_sums = [A[0]]
    for i in range(1, len(A)):
        prefix_sums.append(prefix_sums[-1] + A[i])

    def get_psum(i):
        return 0 if i == -1 else prefix_sums[i]

    # Get min prefix sums
    min_prefix_sums = [prefix_sums[-1]]
    for i in range(len(prefix_sums)-2, -1, -1):
        min_prefix_sums.append(
            min(min_prefix_sums[-1], get_psum(i))
        )
    min_prefix_sums = min_prefix_sums[::-1]

    # Traverse through A
    longest = 0
    i = j = 0
    while i < len(A) and j < len(A):
        if min_prefix_sums[j] - get_psum(i-1) <= k:
            longest = max(longest, j - (i-1))
            j += 1
        else:
            i += 1

    return longest
```

</details>

## Day 23: Querying Range Sums

From [LeetCode](https://leetcode.com/problems/range-sum-query-mutable/):

> Design an array-like data structure that allows you to update elements and query subarray sums efficiently.
>
> ```py
> Input: [1, 3, 5]
>
> Output:
> sumRange(0, 2) # 9
> update(1, 2)
> sumRange(0, 2) # 8
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/querying-range-sums-python/):

```py
class Node:
    def __init__(self, start, last):
        self.start, self.last = start, last
        self.left = self.right = None        
        self.total = 0


class NumArray:
    def __init__(self, nums: List[int]):
        # O(n) time operation for n data entries
        def create(i, j):
            if i > j:
                return None

            elif i == j:
                # Leaf node
                node = Node(i, j)
                node.total = nums[i]
                return node
            
            # Find midpoint to split child ranges by
            m = (i + j) // 2
            node = Node(i, j)
            
            # Construct child nodes and increment
            # the total as we backtrack
            node.left = create(i, m)
            node.right = create(m+1, j)
            node.total = node.left.total + node.right.total
            return node

        self.root = create(0, len(nums)-1)


    def update(self, i: int, val: int) -> None:
        # O(log n) time operation
        def helper(node, i, val):
            if node.start == i and node.last == i:
                diff = val - node.total
                node.total = val
                return diff

            mid = (node.start + node.last) // 2
            diff = (
                helper(node.left, i, val)
                if node.start <= i <= mid
                else helper(node.right, i, val)
            )

            # Update ancestor sums as we backtrack
            node.total += diff
            return diff

        helper(self.root, i, val)


    def sumRange(self, i: int, j: int) -> int:
        # O(log n) time operation
        def helper(node, i, j):
            # Base Case: End-to-end match
            if node.start == i and node.last == j:
                return node.total

            mid = (node.start + node.last) // 2
            
            # Case 1: Whole range falls on right            
            if i <= mid and j <= mid:
                return helper(node.left, i, j)
            
            # Case 2: Whole range falls on left
            elif mid < i and mid < j:
                return helper(node.right, i, j)

            # Case 3: Range is split between both subtrees
            elif i <= mid and mid < j:
                left = helper(node.left, i, mid)
                right = helper(node.right, mid+1, j)
                return left + right

        return helper(self.root, i, j)
```

</details>

## Day 24: Querying Range Sums (2D)

From [LeetCode](https://leetcode.com/problems/range-sum-query-2d-mutable/):

> Design a matrix-like data structure that allows you to update elements and query submatrix sums efficiently.
>
> ```py
> Input: [
>     [3, 0, 1, 4, 2],
>     [5, 6, 3, 2, 1],
>     [1, 2, 0, 1, 5],
>     [4, 1, 0, 1, 7],
>     [1, 0, 3, 0, 5]
> ]
> 
> Output:
> sumRegion(2, 1, 4, 3) # 8
> update(3, 2, 2)
> sumRegion(2, 1, 4, 3) # 10
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/querying-range-sums-2D-python/):

```py
class Node:
    def __init__(self, start, last):
        self.start, self.last = start, last
        self.left = self.right = None
        # Points to integer OR segment tree
        self.v = None


# Utils

def merge_vals(v1, v2):
    def merge_nodes(n1, n2):
        if not n1 and not n2:
            return None

        merged_node = Node(n1.start, n1.last)
        merged_node.left = merge_nodes(n1.left, n2.left)
        merged_node.right = merge_nodes(n1.right, n2.right)
        merged_node.v = n1.v + n2.v
        return merged_node

    if (
        isinstance(v1, SegmentTree) and
        isinstance(v2, SegmentTree)
    ):
        st = SegmentTree()
        st.root = merge_nodes(v1.root, v2.root)
        return st

    else:
        return v1 + v2


    
class SegmentTree:
    def __init__(self):
        self.root = None

    def create(self, ref_data):
        def helper(i, j):
            if i > j:
                return None
            elif i == j:
                node = Node(i, i)
                if isinstance(ref_data[i], list):
                    node.v = SegmentTree()
                    node.v.create(ref_data[i])
                else:
                    node.v = ref_data[i]
                return node

            m = (i + j) // 2
            node = Node(i, j)
            node.left = helper(i, m)
            node.right = helper(m+1, j)
            node.v = merge_vals(node.left.v, node.right.v)
            return node

        self.root = helper(0, len(ref_data)-1)

    def lookup(self, r, c):
        def helper(node, i):
            nonlocal c

            # Leaf case
            if node.start == i and node.last == i:
                if isinstance(node.v, SegmentTree):
                    return helper(node.v.root, c)
                else:
                    return node.v

            # General case
            m = (node.start + node.last) // 2
            if 0 <= i <= m:
                return helper(node.left, i)
            return helper(node.right, i)

        return helper(self.root, r)

    def update(self, r, c, val):
        def helper(node, i, diff):
            nonlocal c
            
            if node:
                if isinstance(node.v, SegmentTree):
                    helper(node.v.root, c, diff)
                else:
                    node.v += diff

                m = (node.start + node.last) // 2
                if 0 <= i <= m:
                    helper(node.left, i, diff)
                else:
                    helper(node.right, i, diff)

        diff = val - self.lookup(r, c)
        return helper(self.root, r, diff)

    
    def query(self, r1, c1, r2, c2):
        def helper(node, i, j):
            nonlocal c1
            nonlocal c2
            
            # Base case: End-to-end match
            if node.start == i and node.last == j:
                if isinstance(node.v, SegmentTree):
                    return helper(node.v.root, c1, c2)
                else:
                    return node.v

            m = (node.start + node.last) // 2

            # Case 1: Query falls on left
            if i <= m and j <= m:
                return helper(node.left, i, j)
            
            # Case 2: Query falls on right
            elif m < i and m < j:
                return helper(node.right, i, j)
            
            # Case 3: Query falls on both sides
            elif i <= m and m < j:
                left = helper(node.left, i, m)
                right = helper(node.right, m+1, j)
                return left + right

        return helper(self.root, r1, r2)


class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.st = SegmentTree()
        self.st.create(matrix)

    def update(self, row: int, col: int, val: int) -> None:
        self.st.update(row, col, val)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.st.query(row1, col1, row2, col2)
```

</details>

## Day 25: The Color Sorting Problem

From [LeetCode](https://leetcode.com/problems/sort-colors/):

> Given an array with `n` objects colored red, white, or blue, sort them in-place in a single pass. Objects of the same color should be adjacent to each other and the sorted colors should be in the order of red, white, and blue.
>
> ```py
> Input:  [2, 0, 2, 1, 1, 0]
> Output: [0, 0, 1, 1, 2, 2]
> ```


<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/thinking-about-partitions-python/):

```py
def sort_colors(A):
    # A[:red] -> Red bucket
    # A[red:white] -> White bucket
    # A[blue:] -> Blue bucket
    # A[white:blue] -> Unseen bucket
    red = 0
    white = 0
    blue = len(A)

    # Navigate through the array until size of 
    # unseen subarray shrinks to zero
    while white < blue:
        if A[white] == 0:
            A[white], A[red] = A[red], A[white]
            red += 1
            white += 1
        
        elif A[white] == 1:
            white += 1
        
        else: # A[white] == 2
            blue -= 1
            A[white], A[blue] = A[blue], A[white]
```

</details>

## Day 26: Huffman Coding

From [EPI](https://github.com/adnanaziz/EPIJudge/blob/master/epi_judge_python/huffman_coding.py):

> Given a set of characters and their corresponding frequencies of occurrence, produce Huffman codes for each character such that the average code length is minimised. Return the average code length.
>
> Note: Huffman codes are prefix codes –– one code cannot be a prefix of another. For example, `"011"` is a prefix of `"0110"` but not a prefix of `"1100"`.
>
> Note: The average code length is defined to be the sum of the product of the length of each character's code word with that character's frequency.
>
> ```py
> Input: [
>     ["a", 8.167], ["b", 1.492], ["c", 2.782], ["d", 4.253],
>     ["e", 12.702], ["f", 2.228], ["g", 2.015], ["h", 6.094],
>     ["i", 6.966], ["j", 0.153], ["k", 0.772], ["l", 4.025],
>     ["m", 2.406], ["n", 6.749], ["o", 7.507], ["p", 1.929],
>     ["q", 0.095], ["r", 5.987], ["s", 6.327], ["t", 9.056],
>     ["u", 2.758], ["v", 0.978], ["w", 2.36], ["x", 0.15],
>     ["y", 1.974], ["z", 0.074]
> ]
>
> Output: 4.205019999999999
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/huffman-coding-python/):

```py
class Node:
    def __init__(self, c=None, freq=None):
        self.c = c  # Character
        self.freq = freq  # Frequency
        self.left = self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_coding(symbols):
    # Store symbols as nodes, heapify nodes
    q = []
    for cwf in symbols:
        heapq.heappush(q, (cwf.freq, Node(cwf.c, cwf.freq)))

    # Bottom-up construction of Huffman tree 
    # Pop nodes / subtrees in pairs and combine them
    while len(q) >= 2:
        node1 = heapq.heappop(q)[1]
        node2 = heapq.heappop(q)[1]
        node3 = Node(c=None, freq=node1.freq + node2.freq)
        node3.left = node1
        node3.right = node2
        heapq.heappush(q, (node3.freq, node3))

    # Generate codewords
    codewords = {}
    def DFS(node, acc_code):
        if not node:
            return None
        if node.c:
            codewords[node.c] = (acc_code, node.freq)
        DFS(node.left, acc_code + '0')
        DFS(node.right, acc_code + '1')
    DFS(q[0][1], '')

    # Calculate avg code length
    res = 0
    for char in codewords:
        code, freq = codewords[char]
        res += (freq / 100) * len(code)
    return res
```

</details>

## Day 27: Convex Hull

From [LeetCode](https://leetcode.com/problems/erect-the-fence/):

> There are multiple trees growing in a garden, each with a distinct `(x, y)` coordinate. You are tasked to enclose all of these trees with a single rope. The amount of rope you use should be minimised. Find the coordinates of the trees located along the rope.
> 
> Note: The garden has at least one tree and the points you're given have no order. You're not required to sort your output either.
>
> ```py
> Input:  [[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]]
> Output: [[1, 1], [2, 0], [4, 2], [3, 3], [2, 4]]
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/convex-hull-python/):

```py
import math
import fractions


def g(ref, p):
    numerator = p[1] - ref[1]
    denominator = p[0] - ref[0]
    if denominator == 0:
        return math.inf
    return fractions.Fraction(numerator, denominator)


def outer_trees(self, points):
    # Minor optimisation
    if len(points) <= 3:
        return points

    def build_chain(comp):
        chain = [points[0]]
        for p in points[1:]:
            while (
                len(chain) >= 2 and
                # Compare the previous gradient to the incoming
                # gradient. If we detect a violation, pop the
                # last node in our chain.
                comp(
                    g(chain[-2], chain[-1]),
                    g(chain[-1], p)
                )
            ):
                chain.pop()
            chain.append(p)
        return chain

    points.sort()
    topchain = build_chain(operator.lt)
    btmchain = build_chain(operator.gt)
    # De-duplicate points in both chains
    return set([tuple(p) for p in topchain + btmchain])

```

</details>

## Day 28: Sliding Puzzle

From [LeetCode](https://leetcode.com/problems/sliding-puzzle/):

> A 2x3 puzzle board has tiles numbered 1 to 5 and an empty square represented by 0. A move consists of swapping 0 with an adjacent number. The puzzle is solved when `[[1, 2, 3], [4, 5, 0]]` is achieved. Given a puzzle configuration, return the least number of moves required to solve it.
> 
> Note: If it is impossible for the puzzle to be solved, return `-1`.
>
> ```py
> Input:  [[1, 2, 3], [4, 0, 5]]
> Output: 1 # Swap 0 and 5.
>
> Input:  [[1, 2, 3], [5, 4, 0]]
> Output: -1 # No number of moves can solve the puzzle. 
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/sliding-puzzle-python/):

```py
NEXT = {
    0: (1, 3),
    1: (0, 2, 4),
    2: (1, 5),
    3: (0, 4),
    4: (1, 3, 5),
    5: (2, 4)
}

def sliding_puzzle(self, board):
    board = [[str(x) for x in row] for row in board]
    board = ''.join(board[0]) + ''.join(board[1])
    frontier = [board]
    seen = set([board])
    moves = 0

    while frontier:
        next_frontier = []

        for b in frontier:
            if b == "123450":
                return moves

            i = b.index('0')
            for j in NEXT[i]:
                A = list(b)
                A[i], A[j] = A[j], A[i]
                nb = ''.join(A)
                if nb not in seen:
                    seen.add(nb)
                    next_frontier.append(nb)

        frontier = next_frontier
        moves += 1

    return -1
```

</details>
