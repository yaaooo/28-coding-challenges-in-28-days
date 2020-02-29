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

Solution based on [this walkthrough](https://yao.page/posts/alien-dictionary-python/):

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
