"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import numpy as np
from itertools import product
import frozendict
from msdm.core.problemclasses.mdp.policy import TabularPolicy
from msdm.core.distributions import DictDistribution

from collections import namedtuple
from functools import lru_cache
SubgraphSolution = namedtuple("SubgraphSolution", "policy value_function initial_value terminates")

# value iteration
def batched_value_iteration(tfs, rfs,dr):
    # Value iteration over a btach of transition functions and rewards
    n_batches = tfs.shape[0]
    n_states = tfs.shape[1]
    
    assert tfs.shape[0:1] == rfs.shape[0:1]
    vs = np.zeros((n_batches, n_states))
    
    terminal_states = (tfs > 0).sum(axis=-1, keepdims=True) == 0
    aa = ((tfs > 0) | terminal_states).astype(float) 
    aa = np.log(aa)
    
    for i in range(10):
        qs = np.einsum("bsn,bn->bsn", tfs, vs) + aa
        new_vs = rfs + dr*qs.max(axis=-1) 
        if (vs == new_vs).all():
            break
        vs = new_vs
    
    pis = (qs == qs.max(axis=-1)[:, :, None]).astype(float) * (~terminal_states).astype(float)# match state with max q state but avoid deadend states 
    pis = pis/pis.sum(axis=-1)[:, :, None]
    pis = np.nan_to_num(pis,nan=0.0) # change nan to 0
    
    return vs, pis

@lru_cache(maxsize=None)
def all_subgraph_solutions(base_mdp_params):
    """Return solutions for all subgraphs of base mdp"""

    # Construct base graph and reward function
    connections = base_mdp_params['connections']
    rf = base_mdp_params['goal_values']
    states = sorted(set(sum(connections, ())))
    initial_states = {s for s in states if s[1] == base_mdp_params['height'] -1}
    initial_state_idx = tuple([si for si, s in enumerate(states) if s in initial_states])
    terminal_states = {s for s in states if s[1] == 0}
    terminal_state_vec = np.array([1 if s in terminal_states else 0 for s in states])
    terminal_state_idx = tuple([si for si, s in enumerate(states) if s in terminal_states])
    
    base_tf = np.zeros((len(states), len(states)))
    base_rf = np.zeros((len(states),))
    for from_s, to_s in connections:
        base_tf[states.index(from_s), states.index(to_s)] = 1
        base_rf[states.index(to_s)] = rf[to_s] + base_mdp_params['step_cost']
    for terminal in terminal_states:
        base_tf[states.index(terminal), :] = 0
    
    # print(states)

    # enumerate all subgraphs
    tfs = []
    rfs = []
    connection_idxs = np.where(base_tf) # index for non zero elements in base_tf
    all_params = []
    for sel in list(product([0, 1], repeat=int(base_tf.sum()))): # bianry permuations based on number of non zero elements in base_tf
        connections = set([])
        sel_tf = np.copy(base_tf)
        for idx, remove in enumerate(sel): # loop through each sel, looks like (i, sel_i)
            i, j = connection_idxs[0][idx], connection_idxs[1][idx] # convert to the index for base_tf
            if remove == 0: # keep it
                connections.add((states[i], states[j])) # add coord states to connection list
                continue
            sel_tf[i, j] = 0 # change one of the the base_tf non zero elements to 0
        if connections==set():
            # print('removed edgeless graph')
            continue
        tfs.append(sel_tf)
        rfs.append(base_rf)
        all_params.append(frozendict.frozendict({**base_mdp_params, "connections": frozenset(connections)}))
    tfs = np.stack(tfs)
    rfs = np.stack(rfs)
    rfs = rfs + (((tfs.max(axis=-1)-1)+terminal_state_vec)*1000)#add deadend cost
    vs, pis = batched_value_iteration(tfs, rfs,base_mdp_params['discount_rate'])
    rand_pis = tfs/(tfs+1e-10).sum(-1, keepdims=True)
    sr = np.linalg.inv(np.eye(len(states))[None, :, :] - (rand_pis*tfs))
    terminating = sr[:, initial_state_idx, terminal_state_idx].sum(-1) > 0 
    
    subgraph_solutions = {}
    for pi, v, terminates, sub_mdp_params in zip(pis, vs, terminating, all_params):
        subgraph_solutions[sub_mdp_params] = SubgraphSolution(
            policy = TabularPolicy(
                {
                    s: DictDistribution({ns : pi[si][sj] for sj, ns in enumerate(states) if pi[si][sj] > 0})
                    for si, s in enumerate(states) 
                    if np.sum(pi[si])>0 # remove empty DictDistribution 

                }),
            value_function = dict(zip(states, v)),
            initial_value = v[initial_state_idx],
            terminates = terminates
        )
        
    return subgraph_solutions

def clear_subgraph_cache():
    all_subgraph_solutions.cache_clear()
    return