"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

from frozendict import frozendict
from operator import itemgetter
from functions.functions import create_mdp
import random
import numpy as np

def sample_teaching_trial(
    true_graph,
    mdp_params,
    seed=None,
    max_tries=1000):
    
    if seed==None:
        print('no seed given')
        assert not seed==None 
        
    rng = random.Random(seed)
    
    def _sample():
        states = set([])
        subgraph = set([])
        for edge in true_graph:
            if rng.random() > .5:
                subgraph.add(edge)
            states.add(edge[0])
            states.add(edge[1])

        goal_values = {}
        for state in states:
            if state == max(states,key=itemgetter(1)):
                goal_values[state] = 0
            else:
                goal_values[state] = rng.randint(0, 3)
                
            # if state[1] == 0 :
            #     goal_values[state] += np.finfo(np.float).eps
                
        return dict(
            goal_values=frozendict(goal_values),
            connections=frozenset(subgraph)
        )
    
    for i in range(max_tries):
        trial_params = _sample()
        sampled_params = frozendict({
            **mdp_params,
            **trial_params,
        })
        mdp = create_mdp(sampled_params)
        if mdp.has_solution():
            return trial_params
    assert False, "Maximum samples reached"

def make_true_graph(height):
    if(height==3):
        true_graph = frozenset({
            ((1,2),(0,1)),
            ((1,2),(1,1)),
            ((1,2),(2,1)),

            ((0,1),(0,0)),
            ((0,1),(1,0)),

            ((1,1),(0,0)),
            ((1,1),(1,0)),
            ((1,1),(2,0)),

            ((2,1),(1,0)),
            ((2,1),(2,0)),
            })
    elif(height==4):
        true_graph = frozenset({
            ((1,3),(0,2)),
            ((1,3),(1,2)),
            ((1,3),(2,2)),

            ((2,2),(1,1)),
            ((2,2),(2,1)),

            ((1,2),(0,1)),
            ((1,2),(1,1)),
            ((1,2),(2,1)),

            ((0,2),(0,1)),
            ((0,2),(1,1)),

            ((0,1),(0,0)),
            ((0,1),(1,0)),

            ((1,1),(0,0)),
            ((1,1),(1,0)),
            ((1,1),(2,0)),

            ((2,1),(1,0)),
            ((2,1),(2,0)),
            })   
    else:
        print('height must 3 or 4')

    return true_graph

def create_random_mdp_params(seed=None,
                  height=None,
                  width=3,
                  goal_values=None,
                  connections=None,
                  step_cost=0.0,
                  discount_rate=1
               ):
    if height==None:
        print('no height given')
        assert not height==None
    if seed==None:
        print('no seed given')
        assert not seed==None    

    mdp_params = dict(
    height= height,
    width=width,
    step_cost=0.0,
    discount_rate=1
    )

    true_graph=make_true_graph(height)

    trial_params=sample_teaching_trial(true_graph,mdp_params,seed)
    
    random_mdp_params = {**mdp_params, 
                     'connections': trial_params.get('connections'),
                     'goal_values': trial_params.get('goal_values')}
    

    return random_mdp_params


def create_true_mdp_params(seed=None,
                  height=None,
                  width=3,
                  goal_values=None,
                  connections=None,
                  step_cost=0.0,
                  discount_rate=1
               ):
    if height==None:
        print('no height given')
        assert not height==None
    if seed==None:
        print('no seed given')
        assert not seed==None    

    mdp_params = dict(
    height= height,
    width=width,
    step_cost=0.0,
    discount_rate=1
    )

    true_graph=make_true_graph(height)

    trial_params=sample_teaching_trial(true_graph,mdp_params,seed)
    
    true_mdp_params = {**mdp_params, 
                     'connections': true_graph,
                     'goal_values': trial_params.get('goal_values')}
    

    return true_mdp_params
