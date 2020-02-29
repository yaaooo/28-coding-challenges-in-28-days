## Day 15: Optimising Highway Networks

From [EPI](https://github.com/adnanaziz/EPIJudge/blob/master/epi_judge_python/road_network.py):

> Write a program which takes an existing highway network (specified as a set of highway sections between pairs of cities) and proposals for new highway sections, and returns the proposed highway section which leads to the most improvement in the total driving distance.
> 
> The total driving distance is defined to be sum of the shortest path distances between all pairs of cities. All sections, existing and proposed, allow for bi-directional traffic, and the original network is connected.
>

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/optimising-highway-networks-python/):

```py
Highway = collections.namedtuple('Highway', ('x', 'y', 'dist'))

def find_best_proposal(H, P, n):

    # Computing existing shortest distances
    M = [[math.inf for _ in range(n)] for _ in range(n)]

    for i in range(n):
        M[i][i] = 0

    for h in H:
        M[h.x][h.y] = M[h.y][h.x] = h.dist

    for k, i, j in itertools.product(range(n), repeat=3):
        M[i][j] = min(M[i][j], M[i][k] + M[k][j])

    # Evaluating proposals
    best_proposal, best_savings = None, None

    for p in P:

        total_savings = 0

        for i, j in itertools.product(range(n), repeat=2):
            savings = M[i][j] - (M[i][p.x] + p.dist + M[p.y][j])
            total_savings += max(savings, 0)

        if not best_savings or total_savings > best_savings:
            best_proposal, best_savings = p, total_savings

    return best_proposal
```

</details>

## Day 16: Kth Smallest Element in Two Sorted Arrays

From [EPI](https://github.com/adnanaziz/EPIJudge/blob/master/epi_judge_python/kth_largest_element_in_two_sorted_arrays.py):

> Given two sorted arrays, find the kth smallest element in the array that would be produced if we combined and sorted them. Array elements may be duplicated within and across the input arrays.
>
> ```py
> Input: 
> A = [-7, -5, 0, 2, 3, 3, 5, 7]
> B = [1, 2, 3]
> k = 3
> 
> Output: 0
> 
> # 0 is smaller than all except for -7 and -5,
> # making it the 3rd smallest element.
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/kth-smallest-element-in-two-sorted-arrays-python/):

```py
def find_kth_smallest(A, B, k):

    lo = max(0, k - len(B))
    hi = min(len(A), k)

    def get_val(arr, i):
        if 0 <= i <= len(arr) - 1:
            return arr[i]
        return math.inf * (-1 if i < 0 else 1)

    while lo <= hi:
        A_size = (lo + hi) // 2
        B_size = k - A_size

        A_left, A_right = get_val(A, A_size-1), get_val(A, A_size)
        B_left, B_right = get_val(B, B_size-1), get_val(B, B_size)

        if A_left <= B_right and B_left <= A_right:
            return max(A_left, B_left)

        elif A_left > B_right:
            lo, hi = lo, A_size - 1

        else:  # B_left > A_right
            lo, hi = A_size + 1, hi
```

</details>

## Day 17: Validating K-Palindromes

From [LeetCode](https://leetcode.com/problems/valid-palindrome-iii/):

> Given a string `s` and an integer `k`, find out if the given string is a k-palindrome or not. A string is a k-palindrome if it can be transformed into a palindrome by removing at most `k` characters from it.
>
> ```py
> Input: s = "abcdeca", k = 2
> Output: True # Remove 'b' and 'e'
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/validating-k-palindromes-python/):

```py
def is_k_palindrome(s, k):
    def delete_dist(a, b):
        DP = [i + 1 for i in range(len(a))]
        
        for j in range(len(b)):
            
            new_DP = DP[:]

            for i in range(len(a)):
                if a[i] == b[j]:
                    new_DP[i] = (
                        # a[:i] -> b[:j]
                        DP[i-1] if i > 0 else j
                    )
                    
                else:
                    new_DP[i] = min(
                        # a[:i] -> b[:j+1], remove a[i]
                        (new_DP[i-1] if i > 0 else j + 1) + 1,
                        # b[:j] -> a[:i+1], remove b[j]
                        DP[i] + 1
                    )
                    
            DP = new_DP
        
        return DP[-1]
    
    return delete_dist(s, s[::-1]) <= 2 * k
```

</details>

## Day 18: The Skyline Problem

From [LeetCode](https://leetcode.com/problems/the-skyline-problem/):

> A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Suppose you are given the locations and height of all the buildings in a cityscape. Write a program to output the skyline formed by these buildings.
>
> The geometric information of each building is represented by a triplet of integers `[Li, Ri, Hi]`, where `Li` and `Ri` are the `x` coordinates of the left and right edge of the ith building and `Hi` is its height.
> 
> The output is a list of "key points" in the format of `[[x1, y1], ...]`. A key point is the left endpoint of a horizontal line segment. Note that last key point, where the rightmost building ends, is merely used to mark the termination of the skyline, and always has zero height.
>
> Also, the ground in between any two adjacent buildings should be considered part of the skyline contour.
>
> ```py
> Input: [
>     [2, 9, 10], [3, 7, 15], [5, 12, 12],
>     [15, 20, 10], [19, 24, 8]
> ]
> Output: [
>     [2, 10], [3, 15], [7, 12], [12, 0],
>     [15, 10], [20, 8], [24, 0]
> ]
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/the-skyline-problem-python/):

```py
import collections

Building = collections.namedtuple('Building', ('l', 'r', 'h'))
Active = collections.namedtuple('Active', ('val', 'b'))

def get_skyline(buildings):
    scan = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    B = []
    for b_idx, b in enumerate(buildings):
        l, r, h = b
        B.append(Building(l, r, h))
        scan[l]['left'].append(b_idx)
        scan[r]['right'].append(b_idx)

    active = []
    res = []
    scan = sorted(scan.items())

    for x, data in scan:
        while active and active[0].b.r <= x:
            heapq.heappop(active)

        for b_idx in data['left']:
            heapq.heappush(
                active,
                Active(-B[b_idx].h, B[b_idx])
            )

        if not res:
            res.append((x, active[0].b.h))
        elif not active:
            res.append((x, 0))
        elif active[0].b.h != res[-1][1]:
            res.append((x, active[0].b.h))

    return res
```

</details>

## Day 19: Regular Expression Matching

From [LeetCode](https://leetcode.com/problems/regular-expression-matching/):

> Given an input string `s` and a pattern `p`, implement regular expression matching with support for `'.'` and `'*'`.
>
> `'.'` Matches any single character.
>
> `'*'` Matches zero or more of the preceding element.
>
> The matching should cover the entire input string (not partial).
>  
> ```py
> Input: s = "aa", p = "a"
> Output: False
> ```
> ```py
> Input: s = "aa", p = "a*"
> Output: True
> ```


<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/regular-expression-matching-python/):

```py
def is_match(s, p):

    n, m = len(s), len(p)
    
    # Initialise DP array
    DP = [
        [False for _ in range(m+1)]
        for _ in range(n+1)
    ]

    # Empty substring vs empty pattern
    DP[0][0] = True

    # Empty substring vs non-empty subpatterns
    for j in range(2, m+1, 2):
        if p[j-1] == '*':
            DP[0][j] |= DP[0][j-2]

    # Non-empty substrings vs non-empty subpatterns
    for i in range(1, n+1):
        for j in range(1, m+1):

            # Case 1: Matching tails
            # (x matches x -> xy matches xy)
            if is_eq(p[j-1], s[i-1]):
                DP[i][j] |= DP[i-1][j-1]

            # Case 2: With '*' Quantifier
            if p[j-1] == '*':

                # Case 2A: Single instance quantifier
                # (x matches x -> x* matches x)
                DP[i][j] |= DP[i][j-1]
                
                # Consider character-quantifier pairs
                if j >= 2:
                    
                    # Case 2B: Multiple instance quantifier
                    # (x* matches x -> x* matches xx)
                    if is_eq(p[j-2], s[i-1]):
                        DP[i][j] |= DP[i-1][j]

                    # Case 2C: Zero instance quantifier
                    # (x matches x -> xy* matches x)
                    DP[i][j] |= DP[i][j-2]

    return DP[-1][-1]
```

</details>

## Day 20: Detecting Arbitrage in Foreign Exchange Markets

From [EPI](https://github.com/adnanaziz/EPIJudge/blob/master/epi_judge_python/arbitrage.py):

> Suppose you are given a 2D matrix representing exchange rates between currencies. You want to determine if arbitrage exists in the market (i.e. if there is a way to start with a single unit of some currency C and convert it back to more than one unit of C through a sequence of exchanges).
>
> ```py
> Input: [
>     [1.0,  2.0, 1.0],
>     [0.5,  1.0, 4.0],
>     [1.0, 0.25, 1.0]
> ]
> Output: True
>
> # Arbitrage Path:
> # Value (idx)
> # 1.0 (0) -> 2.0 (1) -> 8.0 (2) -> 8.0 (0)
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/detecting-arbitrage-in-foreign-exchange-markets-python/):

```py
def has_arbitrage(M):
    
    n = len(M)

    # Apply log on edges, flip signs
    for i, j in itertools.product(range(n), repeat=2):
        M[i][j] = -math.log2(M[i][j])

    # Track shortest "dist" from source for each node
    D = [math.inf for _ in range(n)]
    D[0] = 0
 
    # Run Bellman-Ford
    for _ in range(n):
        for i, j in itertools.product(range(n), repeat=2):
            D[j] = min(D[j], D[i] + M[i][j])
    
    # Check for -ve cycles
    for i, j in itertools.product(range(n), repeat=2):
        if D[i] + M[i][j] < D[j]:
            return True

    return False
```

</details>

## Day 21: Smaller Numbers After Self

From [LeetCode](https://leetcode.com/problems/count-of-smaller-numbers-after-self/):

> You are given an integer array `A` and you have to return a new `counts` array. The counts array has the property where `counts[i]` is the number of smaller elements to the right of `A[i]`.
>
> ```py
> Input: [5, 2, 6, 1]
> Output: [2, 1, 1, 0]
> 
> # Explanation:
> # To the right of 5 there are 2 smaller elements (2 and 1).
> # To the right of 2 there is only 1 smaller element (1).
> # To the right of 6 there is 1 smaller element (1).
> # To the right of 1 there is 0 smaller element.
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/smaller-numbers-after-self-python/):

```py
import collections

Entry = collections.namedtuple('Entry', ('val', 'prev_i'))

def count_smaller(A):

    n = len(A)

    # Output array, keeps track of counts for each index
    counts = [0 for _ in range(n)]

    # Attach original index to each value
    A = [Entry(val, i) for i, val in enumerate(A)]
    
    # Define merge algo
    def merge(L1, L2):
        i = j = 0
        L3 = []

        while i < len(L1) and j < len(L2):
            if L1[i].val <= L2[j].val:
                # Index j captures the number of smaller
                # elements found on the right
                counts[L1[i].prev_i] += j
                L3.append(L1[i])
                i += 1
                
            else: # L1[i].val > L2[j].val
                L3.append(L2[j])
                j += 1

        while i < len(L1):
            counts[L1[i].prev_i] += j
            L3.append(L1[i])
            i += 1
        
        while j < len(L2):
            L3.append(L2[j])
            j += 1
                            
        return L3
    
    # Define mergesort algo
    def msort(s, l):
        if s > l:
            return []
        elif s == l:
            return [A[s]]

        m = (s + l) // 2
        return merge(msort(s, m), msort(m+1, l))

    # Run mergesort
    msort(0, n-1) 
    return counts
```

</details>
