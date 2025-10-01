"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import numpy    as np
import scipy    as sp
import matplotlib    as mpl
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gd
from matplotlib import pyplot
import random


myred = mpl.cm.Reds(np.linspace(0,1,6))
myred = mpl.colors.ListedColormap(myred[:-2:,:-1])

pyplot.style.use('default')

import itertools
from itertools import combinations
from frozendict import frozendict
from collections import defaultdict
from collections import Counter
from msdm.algorithms import PolicyIteration, ValueIteration
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from functions.utils import CustomDictDistribution as DictDistribution
from functions.graphWorld import GraphWorld, GraphWorldPlotter,intTocoord
from functions.fitting import get_policy

# ======================================================= #
# New stuff for refactoring

from functools import lru_cache
from joblib import Memory
from frozendict import frozendict

from msdm.algorithms import PolicyIteration
cachedir = "_cache"
memory = Memory(cachedir, verbose=0)

# @lru_cache(maxsize=int(1e5))
def create_mdp(params):
    return GraphWorld(**params)

# @lru_cache(maxsize=int(1e5))
def solve_mdp(params):
    mdp = create_mdp(params)
    plan_res = PolicyIteration().plan_on(mdp)
    return plan_res

# @lru_cache(maxsize=int(1e5))
# @memory.cache #careful with this!
def calc_teacher_graph_prior(true_mdp_params):
    """Teacher's prior over learner's graphs"""
    true_mdp = create_mdp(true_mdp_params)
    #TODO: make worldmaker logic (graph enumeration) part of MDP
    graphs = [frozenset(g) for g in worldmaker(true_mdp)]
    graph_dist = DictDistribution.uniform(graphs)
    
    # restrict to solvable graphs
    def is_valid(connections):
        new_params = frozendict({**true_mdp_params, "connections": connections})
        mdp = create_mdp(new_params)
        return mdp.has_solution()
    graph_dist = graph_dist.condition(is_valid)
    return graph_dist

def powerset(S, maxsize=float('inf'), minsize=0):
    for n in range(minsize, len(S) + 1):
        if n > maxsize:
            break
        for c in combinations(S, n):
            yield set(c)

def successor_possibilities(s,mdp):
    #make sure the weight of mdp 
    #use list(itertools.product(*[successor_possibilities(s,tot) for s in tot.state_list])) to find all the permutaitons.
    
    x,y = s
    #special case of end
    if y == 0:
        successors = [set()]
    #special case all nodes in height 1 points to last node
    else:
        successors = [set()]
        #compute possible successor states
        possible_successor_states = [(nx, y - 1) for nx in range(mdp.width) if max(0, x-1) <= nx <= x+1]
        
        successors=list(powerset(possible_successor_states))
        
        #append those states to successors
        # [successors.append([possible_successor_states[i-1] for i in index[j] if i != 0]) for j in range(len(index))]
        #append the diffrent combinations 
             
    return successors

def worldmaker(mdp):
    # worlds_temp is set of all worlds but this is only second half of the conneciton (successors)
    worlds_temp = itertools.product(*[successor_possibilities(s,mdp) for s in mdp.state_list])
    
    for w in worlds_temp:
        connections = set()
        #connect each world outcome to the state, so connect the successors to the predecessor 
        for s,successors in zip(mdp.state_list,w):
            #this is so if there are multiple successors for the same predecessor 
            for ns in successors:
                connections.add((s,ns))
        yield connections

def resptograph_coord(graphSeed,preprocessed):
    
    resptograph_int = list(preprocessed[preprocessed["seed"]==graphSeed]["edge"])

    resptograph = []

    for i in range(0,len(resptograph_int)):
        resptograph.append((intTocoord(resptograph_int[i][0]),intTocoord(resptograph_int[i][1])))
    return resptograph

def resptograph_count(graphSeed,preprocessed):
    resptograph = resptograph_coord(graphSeed,preprocessed)
   
    resp_count= Counter(resptograph)
    for k in intTocoord(preprocessed.at[0,'OBM_AU']):
        if k not in resp_count:
            resp_count[k] = 0
            
    return resp_count


def matching_count(graphSeed,preprocessed):
    matching = []

    for s in preprocessed.subjID.unique().tolist():
        matching.append(
        preprocessed['edge'].where((preprocessed.subjID==s) & (preprocessed.flipped=='FALSE') & (preprocessed.seed==graphSeed)).dropna().tolist() == \
        preprocessed['edge'].where((preprocessed.subjID==s) & (preprocessed.flipped=='TRUE') & (preprocessed.seed==graphSeed)).dropna().tolist()\
        )

    return Counter(matching)

def softmaxSim(preprocessed,fitSummery,seed,model):
    ## it was late this was fast code

    if model == 'OBM_AU':
        invtempModel= 'invtemp_star_OBM'
    elif model == 'POM_AU':
        invtempModel='invtemp_star_POM'
    elif model == 'NBM_AU':
        invtempModel='invtemp_star_NBM'       
        
    
    subjlist= preprocessed.subjID.unique()
    prob = defaultdict(float)
    sumprobs = defaultdict(float)
    sumprobs_int = defaultdict(float)
    
    for subj in subjlist:

        trial = preprocessed.where(preprocessed.subjID==subj).dropna()
        
        advices = trial[model].where(trial.seed==seed).dropna().tolist()
        invtemp = fitSummery[invtempModel].where(fitSummery.subjID==subj).dropna().astype(float)
        # invtemp = 100
        
        if advices== []:
            continue
        elif len(advices)==1:
            left = get_policy(advices[0],invtemp)
            prob[subj] = {k: left[k] for k in set(left)}
        else:
            left = get_policy(advices[0],invtemp)
            right = get_policy(advices[1],invtemp)

            prob[subj] = {k: (left[k] + right[k])/2 for k in set(left)}

    sumprobs = defaultdict(float)
    sumprobs_int = defaultdict(float)
    for key in prob['s0'].keys():
        for subj in subjlist:
            if subj in prob:
                sumprobs[key] += float(prob[subj][key])
        sumprobs_int[(intTocoord(key[0]),intTocoord(key[1]))]= sumprobs[key]/100

    return sumprobs_int

def softmaxSim_exp2(preprocessed,fitSummery,trialtype,model):
    ## it was late this was fast code

    if model == 'U_obm':
        invtempModel= 'invtemp_star_OBM'
    elif model == 'U_fr':
        invtempModel='invtemp_star_OBM'
        
    
    subjlist= preprocessed.subjID.unique()
    prob = defaultdict(float)
    sumprobs = defaultdict(float)
    sumprobs_int = defaultdict(float)
    print(trialtype)
    for subj in subjlist:

        trial = preprocessed.where(preprocessed.subjID==subj).dropna()
        advices = trial[model].where(trial.trialtype==trialtype).dropna().tolist()
        print(subj)
        invtemp = fitSummery[invtempModel].where(fitSummery.subjID==subj).dropna().astype(float)
        print(list(advices))
        left = get_policy(advices[0],invtemp)
        right = get_policy(advices[1],invtemp)

        prob[subj] = {k: (left[k] + right[k])/2 for k in set(left)}

    sumprobs = defaultdict(float)
    sumprobs_int = defaultdict(float)
    for key in prob['s0'].keys():
        for subj in subjlist:
            sumprobs[key] += float(prob[subj][key])
        sumprobs_int[(intTocoord(key[0]),intTocoord(key[1]))]= sumprobs[key]/100

    return sumprobs_int

def softmaxSim_exp2v2(preprocessed,fitSummery,trialtype,model):
    
    def skip_missing_trial(preprocessed,subj,trialtype):
        return preprocessed[(preprocessed.subjID==subj)&(preprocessed.trial_id==trialtype)].dropna().empty

    if model == 'U_obm':
        invtempModel= 'invtemp_star_OBM'
    elif model == 'U_fr':
        invtempModel='invtemp_star_OBM'
        
    
    subjlist= preprocessed.subjID.unique()
    prob = defaultdict(float)
    sumprobs = defaultdict(float)
    sumprobs_int = defaultdict(float)
    for subj in subjlist:
        
        ### this subj doesnt have this trial contitnues
        
        trial = preprocessed.where(preprocessed.subjID==subj).dropna()
        
        if skip_missing_trial(preprocessed,subj,trialtype):
            continue
        else:
            advices = trial[model].where(trial.trialtype==trialtype).dropna().tolist()
            invtemp = fitSummery[invtempModel].where(fitSummery.subjID==subj).dropna().astype(float)
            prob[subj] = get_policy(advices[0],invtemp)

    sumprobs = defaultdict(float)
    sumprobs_int = defaultdict(float)
    
    max_keys = 0
    row_with_max_keys = None

    # Iterate over the dictionary
    for row, data in prob.items():
        # Count the number of keys in the current row
        num_keys = len(data.keys())

        # Check if the current row has more keys than the previous maximum
        if num_keys > max_keys:
            max_keys = num_keys
            row_with_max_keys = row



    for key in prob[row_with_max_keys].keys():
        for subj in subjlist:
            if skip_missing_trial(preprocessed,subj,trialtype):
                continue
            else:
                sumprobs[key] += float(prob[subj][key])
        sumprobs_int[(intTocoord(key[0]),intTocoord(key[1]))]= sumprobs[key]/100

    return sumprobs_int


# helper
def max_value_keys(my_dict,value = 'max'):
    if value == 'max':
        max_value = max(my_dict.values())
    if value == 'min':
        max_value = min(my_dict.values()) 
    max_key_list = [key for key, value in filter(lambda item: item[1] == max_value, my_dict.items())]
    return max_key_list

def dict_values_to_int(x):
    total_value = np.sum(list(x.values()))
    new_x = dict()
    for i in x:
        new_x[i]= round(x[i]/total_value*1000)
    return new_x
