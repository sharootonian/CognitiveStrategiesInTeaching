"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

from collections import defaultdict

def q_values(q_values, state_action_pairs):
    result = {}
    for state, action in state_action_pairs:
        try:
            result[(state, action)] = q_values[state][action]
        except KeyError:
            result[(state, action)] = None  
    return result

def pathAvgUtility(edges, rewards):

    # Build adjacency and identify start
    adj = defaultdict(list)
    nodes = set()
    for u, v in edges:
        adj[u].append(v)
        nodes |= {u, v}
    start = next(n for n in nodes if n not in {v for _, v in edges})

    # DFS to collect all simple paths
    paths = []
    def dfs(node, path):
        path.append(node)
        if node not in adj:
            paths.append(path.copy())
        else:
            for nbr in adj[node]:
                dfs(nbr, path)
        path.pop()

    dfs(start, [])

    # Average path gains per edge
    edge_vals = defaultdict(list)
    for p in paths:
        total = sum(rewards.get(n, 0) for n in p)
        for u, v in zip(p, p[1:]):
            edge_vals[(u, v)].append(total)

    return {e: sum(vals)/len(vals) for e, vals in edge_vals.items()}
