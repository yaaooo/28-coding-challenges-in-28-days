## Day 01: Alien Dictionary

From [LeetCode](https://leetcode.com/problems/alien-dictionary/):

> There is a new alien language which uses the latin alphabet. However, the order among letters is unknown to you. You receive a list of non-empty words from the dictionary, where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.
>
> ```py
> Input: ["wrt", "wrf", "er", "ett", "rftt"]
> Output: "wertf"
> ``` 
> ```py
> Input: ["z","x","z"]
> Output: "" # No valid ordering
> ``` 

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/alien-dictionary-python/):

```py
def alien_dictionary(words):
    n = len(words)

    # Init nodes
    nodes = {}
    for word in words:
        for c in word:
            if c not in nodes:
                nodes[c] = Node(c)

    # Build graph
    for i in range(n-1):
        for c1, c2 in zip(words[i], words[i+1]):
            if c1 == c2:
                continue
            elif c1 != c2:
                nodes[c2].to.add(c1)
                break

    for label in nodes:
        print(label, nodes[label].to)

    # Run topo sort
    visiting = set()
    res = []

    def topo_sort(label):
        visiting.add(label)

        node = nodes[label]
        for v_label in node.to:
            if v_label in visiting:
                return False

            if v_label in nodes:
                if topo_sort(v_label) is False:
                    return False

        res.append(label)
        del nodes[label]

        visiting.remove(label)

    while nodes:
        label = next(iter(nodes))
        if topo_sort(label) is False:
            return ""

    return ''.join(res)

```

</details>


## Day 02: Binary Tree Path Sums

From [LeetCode](https://leetcode.com/problems/path-sum-iii/):

> You are given a binary tree in which each node contains an integer value.
> Find the number of paths that sum to a given value.
> The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).
>
>```py
> Input:
> """
>       10
>      /  \ 
>     5   -3
>    / \    \ 
>   3   2   11
>  / \   \ 
> 3  -2   1
> """
> 8
> 
> Output:
> 3
> # 1: 1 -> 5 -> 3
> # 2: 5 -> 2 -> 1
> # 3: -3 -> 11

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/binary-tree-path-sums-python/):

```py
class Node:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right


def get_num_of_matching_paths(tree, target_sum):
    res = 0
    pathsums = collections.defaultdict(int)

    def traverse(subtree, curr_sum=0):
        nonlocal target_sum
        nonlocal res

        if not subtree:
            return

        curr_sum += subtree.val

        if curr_sum == target_sum:
            res += 1

        if curr_sum - target_sum in pathsums:
            res += pathsums[curr_sum-target_sum]

        pathsums[curr_sum] += 1

        traverse(subtree.left, curr_sum)
        traverse(subtree.right, curr_sum)

        pathsums[curr_sum] -= 1

    traverse(tree)
    return res
```

</details>

## Day 03: Merging Binary Search Trees

From [EPI](https://github.com/adnanaziz/EPIJudge/blob/master/epi_judge_python/bst_merge.py):

> Merge two binary search trees into a single binary search tree.
> 
> ```py
> Input:
> """
>   5
>  / \ 
> 3   7
> 
>   4
>  / \ 
> 2   6
> """
> 
> Output:
> """
>      5
>    /   \ 
>   3     7
>  / \   / 
> 2   4 6
> """

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/merging-binary-search-trees-python/):


```py
def convert_bst_to_list(tree):
    def traverse(subtree):
        if not subtree:
            return None, None

        head = tail = subtree
        if subtree.left:
            left_head, left_tail = traverse(subtree.left)
            head = left_head
            # left_tail <-> subtree
            left_tail.right, subtree.left = subtree, left_tail

        if subtree.right:
            right_head, right_tail = traverse(subtree.right)
            tail = right_tail
            # subtree <-> right_head
            subtree.right, right_head.left = right_head, subtree

        head.left = tail.right = None
        return head, tail

    head, tail = traverse(tree)
    return head


def merge_lists(L1, L2):

    runner1 = L1
    runner2 = L2
    head = runner3 = Node()

    while runner1 and runner2:
        if runner1.data <= runner2.data:
            runner1_right = runner1.right

            runner3.right = runner1
            runner1.left, runner1.right = runner3, None

            runner1 = runner1_right

        else:
            runner2_right = runner2.right

            runner3.right = runner2
            runner2.left, runner2.right = runner3, None

            runner2 = runner2_right

        runner3 = runner3.right

    if runner1:
        runner3.right = runner1
        runner1.left = runner3

    elif runner2:
        runner3.right = runner2
        runner2.left = runner3

    return head.right


def convert_list_to_bst(L):

    length = 0
    runner = L
    while runner:
        length += 1
        runner = runner.right

    runner = L

    def traverse(start, last):
        nonlocal runner

        if not start <= last:
            return None

        mid = (start + last + 1) // 2
        left = traverse(start, mid-1)

        node = runner
        runner = runner.right

        right = traverse(mid+1, last)

        node.left, node.right = left, right
        return node

    return traverse(0, length-1)


def merge_two_bsts(T1, T2):
    L1 = convert_bst_to_list(T1)
    L2 = convert_bst_to_list(T2)
    L3 = merge_lists(L1, L2)
    return convert_list_to_bst(L3)

```

</details>

## Day 04: Minimum Palindromic Partitions

From [Daily Coding Problem](https://www.dailycodingproblem.com/):

> Given a string, split it into as few strings as possible such that each string is a palindrome. For example, given the input string `"racecarannakayak"`, return `["racecar", "anna", "kayak"]`.
> 
> ```py
> Input: "racecarannakayak"
> Output: ["racecar", "anna", "kayak"]
> ```

<details><summary>Answer</summary>


View [walkthrough](https://yao.page/posts/minimum-palindromic-partitions-python/):

```py
import collections

Entry = collections.namedtuple('Entry', ('start', 'count'))

def partition(s):
    DP = [Entry(i, i+1) for i in range(len(s))]
    
    def expand(left, right):
        nonlocal s
        
        if s[left] != s[right]:
            return
        
        while 0 <= left and right <= len(s) - 1:
            if s[left] == s[right]:
                entry = Entry(
                    left,
                    DP[left-1].count + 1 if left - 1 >= 0 else 1
                )

                DP[right] = min([
                    DP[right], entry
                ], key=lambda e: e.count)

                left -= 1
                right += 1
            else:
                break
        
    for i in range(len(s)):
        expand(i, i)
        if i + 1 <= len(s) - 1:
            expand(i, i+1)
    
    
    res, i = [], len(s)-1
    
    while i >= 0:
        entry = DP[i]
        res.append(s[entry.start:i+1])
        i = entry.start - 1
    
    return res[::-1]
```

</details>

## Day 05: Maximum Points on a Line

From [LeetCode](https://leetcode.com/problems/max-points-on-a-line/):

> Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
> ```py
> Input: [[1, 1], [2, 2], [3, 3]]
> Output: 3
> """
> Explanation:
> ^
> |
> |        o
> |     o
> |  o  
> +------------->
> 0  1  2  3  4
> """
> ```
> ```py
> Input: [[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]
> Output: 4
> """
> Explanation:
> ^
> |
> |  o
> |     o        o
> |        o
> |  o        o
> +------------------->
> 0  1  2  3  4  5  6
> """

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/maximum-points-on-a-line-python/):

```py
def max_points(points):

    n = len(points)
    global_max = 0

    for i in range(n):

        lines = collections.defaultdict(int)
        overlaps = 1
        local_max = 1

        for j in range(i+1, n):
            dy = points[j][1] - points[i][1]
            dx = points[j][0] - points[i][0]

            if dy == 0 and dx == 0:
                overlaps += 1
                local_max += 1
                
            else:
                gradient = None
                if dy == 0:
                    gradient = Gradient(0, 1)
                elif dx == 0:
                    gradient = Gradient(1, 0)
                else:
                    if dx < 0:
                        dy, dx = -dy, -dx
                    gcd = math.gcd(dy, dx)
                    gradient = Gradient(dy / gcd, dx / gcd)

                lines[gradient] += 1
                local_max = max(local_max, lines[gradient] + overlaps)

        global_max = max(global_max, local_max)

    return global_max
```

</details>


## Day 06: Longest Non-Contiguous, Non-Decreasing Subsequence

From [EPI](https://github.com/adnanaziz/EPIJudge/blob/master/epi_judge_python/longest_nondecreasing_subsequence.py):

> Find the longest non-contiguous, non-decreasing subsequence in an array of numbers.
> ```py
> Input: [4, 0, 5, 5, 7, 6, 7]
> Output: 5 # Example: 4 5 5 7 7
> ```

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/longest-noncontiguous-nondecreasing-subsequence-python/):

```py
def longest_nondecreasing_subsequence_length(A):
    n = len(A)
    SL = sortedcontainers.SortedList(key=lambda e: e.last_val)

    for x in A:
        new_entry = ActiveSeq(x, 1)
        i = SL.bisect_right(new_entry)
        if i == 0:
            SL.add(new_entry)
        else:
            prev_entry = SL[i-1]
            new_entry = ActiveSeq(x, prev_entry.length + 1)
            if prev_entry.last_val == x:
                SL.remove(prev_entry)
            SL.add(new_entry)

        new_entry_idx = SL.index(new_entry)
        while (
            len(SL) >= new_entry_idx + 2 and
            SL[new_entry_idx+1].length <= new_entry.length
        ):
            SL.remove(SL[new_entry_idx+1])

    return SL[-1].length
```


</details>

## Day 07: Reversing Linked List Nodes in Groups of K

From [LeetCode](https://leetcode.com/problems/reverse-nodes-in-k-group/):

> Reverse the nodes of a linked list, `k` at a time, and return the modified list.
> 
> `k` is a positive integer. It is less than or equal to the length of the linked list. If the number of nodes is not a multiple of `k`, any left-out nodes at the end should remain as they are.
> 
> ```py
> Input:  head = 1 -> 2 -> 3 -> 4 -> 5, k = 2
> Output: 2 -> 1 -> 4 -> 3 -> 5
>
> Input:  head = 1 -> 2 -> 3 -> 4 -> 5, k = 3
> Output: 3 -> 2 -> 1 -> 4 -> 5
> ```
> Notes:
> - Only constant extra memory is allowed.
> - You may not alter the values in the list's nodes, only nodes themselves may be changed.

<details><summary>Answer</summary>

View [walkthrough](https://yao.page/posts/reversing-linked-list-nodes-in-groups-of-k-python/):


```py
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def reverse_list_in_groups_of_k(head, k):
    runner = sentinel = ListNode(None)
    sentinel.next = head
    
    # Get length
    length = 0
    while runner.next:
        runner = runner.next 
        length += 1
    
    # Determine number of groups
    num_groups = length // k

    # Reset runner
    runner = sentinel
        
    for _ in range(num_groups):
        prehead = runner
        backrunner, frontrunner = runner.next, runner.next.next
        
        for _ in range(k-1):
            frontrunner_next = frontrunner.next
            frontrunner.next = backrunner
            backrunner, frontrunner = frontrunner, frontrunner_next
        

        prehead.next.next = frontrunner
        runner = prehead.next
        prehead.next = backrunner
    
    return sentinel.next   
```

</details>
