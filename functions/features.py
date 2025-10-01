"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from frozendict import frozendict
from functions.graphWorld import GraphWorld,coordToint, intTocoord
from functions.functions import max_value_keys,resptograph_count,matching_count,solve_mdp,create_mdp
from functions.mdp_params import make_true_graph,create_random_mdp_params,create_true_mdp_params
from msdm.core.distributions import SoftmaxDistribution
from functions.utils import CustomDictDistribution as DictDistribution
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
def get_feature_levels(true_graph):
    return {(x,y):y[1]+1 for x,y in true_graph}
def get_feature_reward_sum(reward,true_graph):
    return {(x,y):reward[x]+reward[y] for x,y in true_graph}
def get_feature_reward_sum_rmTraj(reward,true_graph,traj):
    traj_edges = [(traj[e],traj[e+1]) for e in range(len(traj)-1)]
    return {(x,y):reward[x]+reward[y] if (x,y) not in traj_edges else 0 for x,y in true_graph}
def get_feature_reward_max(reward,true_graph):
    return {(x,y):max([reward[x],reward[y]]) for x,y in true_graph}
def get_feature_reward_2ndNode_rmTraj(reward,true_graph,traj):
    traj_edges = [(traj[e],traj[e+1]) for e in range(len(traj)-1)]
    return {(x,y):reward[y] if (x,y) not in traj_edges else 0 for x,y in true_graph}
def get_feature_distance_FW(traj,true_graph):
    traj_int = coordToint(traj)
    true_graph_int = coordToint(true_graph)
    # A = floyd_warshall_distance(make_directed_transMat(true_graph_int))
    graph = csr_matrix(make_directed_transMat(true_graph_int))
    A = floyd_warshall(csgraph=graph, directed=True, return_predecessors=False)
    edge_dist = {}
    for s0,s1 in true_graph_int:
        edge = (intTocoord(s0),intTocoord(s1))
        dist = min(A[s0,traj_int].min(), A[traj_int,s1].min())+1
        if dist > 10:
            dist = None
        edge_dist[edge] = dist
    return edge_dist

def get_feature_opt_traj_edge(seed,rewards):
    teacher = create_true_mdp_params(seed,height = 4)
    teacher['goal_values']= rewards 
    # ## NOT correct | we dont want states we want edges
    # sr = solve_mdp(teacher).policy.evaluate_on(GraphWorld(**teacher)).successor_representation
    # # if prob > 0 then some chance to transition to that state based on opt policy 
    # opt_traj_states = [key for key,value in zip(sr.keys(),sr.values()) if value > 0]
    
    ## HACK just run on policy and get traj
    results = solve_mdp(teacher).policy.run_on(GraphWorld(**teacher))
    opt_edges = [(s,ns) for s,ns in zip(results.state_traj,results.action_traj)]
    # to loop and create dict
    true_graph = teacher['connections']
    return {(x,y):1 if (x,y) in opt_edges else 0 for x,y in true_graph}
    

def make_directed_transMat(listotuple):
    INF = 99
    
    if type(listotuple[0]) == tuple:
        A = np.zeros(shape=(10, 10))+INF

        for x,y in listotuple:
            if y<7:
                A[x,x] = A[y,y] = 0
                A[y,x] = 1
                A[x,y] = 1
        return A
    else:
        print("make sure input is a list of tuples")

def floyd_warshall_distance(A):
    for k in range(len(A)): 
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i,j] > A[i,k] + A[k,j]:
                    A[i,j] = A[i,k] + A[k,j]               
    return A

def graph_feature_from_edge_feature(
    edge_feature_func,
    aggregation_func = np.mean
):
    """
    Graph feature function factory -
    Creates a graph feature function based on edge feature
    function and aggregator.
    """
    def graph_feature_function(mdp_params, traj=None):
        edge_feature_vals = edge_feature_func(mdp_params, traj=traj)
        edge_feature_vals_filtered = [v for v in edge_feature_vals.values() if v is not None]
        if edge_feature_vals_filtered == []:
            edge_feature_vals_filtered = 0
            
        return aggregation_func(edge_feature_vals_filtered)
    return graph_feature_function

@lru_cache(maxsize=None)
def graph_score(
    edge_feature_functions,
    feature_weights,
    traj,
    mdp_params 
):
    graph_features = calc_graph_features(
        edge_feature_functions,
        traj,
        mdp_params 
    )
    
    return sum([feature_weights[f]*graph_features[f] for f in graph_features.keys()])

@lru_cache(maxsize=None)
def calc_graph_features(
    edge_feature_functions,
    traj,
    mdp_params 
):
    graph_feature_functions = {}
    for func_name, edge_feature_func in edge_feature_functions.items():
        graph_feature_functions[func_name] = graph_feature_from_edge_feature(edge_feature_func)

    graph_features = {func_name: func(mdp_params, traj) for func_name, func in graph_feature_functions.items()}
    
    return graph_features
    
def clear_graph_score():
    graph_score.cache_clear()
    return

def clear_calc_graph_features():
    calc_graph_features.cache_clear()
    return

edge_feature_functions = dict(
    feature_levels = lambda mdp_params, traj : get_feature_levels(mdp_params['connections']),
    feature_reward = lambda mdp_params, traj: get_feature_reward_sum(mdp_params['goal_values'], mdp_params['connections']), 
    # feature_distance = lambda mdp_params, traj: get_feature_distance_FW(traj, mdp_params['connections'])
)

@lru_cache(maxsize=None)
def is_valid(mdp_params):
    mdp = create_mdp(mdp_params)
    return mdp.has_solution()

def calc_heuristic_teacher_graph_prior(graph_prior, traj,feature_weights):
    subgraph_scores = {}   
    for mdp_params in graph_prior:
        if not is_valid(mdp_params):
            continue  
            
        subgraph_scores[mdp_params['connections']] = graph_score(
            frozendict(edge_feature_functions), 
            frozendict(feature_weights), 
            traj, 
            frozendict(mdp_params)
        )

    graph_dist = DictDistribution(SoftmaxDistribution(subgraph_scores))
    return graph_dist

def calc_obm_teacher_graph_prior(graph_prior):
    subgraph_scores = {}   
    for mdp_params in graph_prior:
        if not is_valid(mdp_params):
            continue  
            
        subgraph_scores[mdp_params['connections']] = 1

    graph_dist = DictDistribution(SoftmaxDistribution(subgraph_scores))
    return graph_dist
