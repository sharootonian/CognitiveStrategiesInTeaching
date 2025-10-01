"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import random
import numpy    as np
import scipy    as sp
import pandas   as pd
import matplotlib    as mpl
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gd
from matplotlib import pyplot
from frozendict import frozendict
from msdm.algorithms import PolicyIteration, ValueIteration
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.distributions import DictDistribution
from matplotlib.collections import LineCollection

myred = mpl.cm.Reds(np.linspace(0,1,6))
myred = mpl.colors.ListedColormap(myred[:-2:,:-1])


pyplot.style.use('default')

class GraphWorld(TabularMarkovDecisionProcess):
    def __init__(self, height, width, connections, goal_values, step_cost, discount_rate):
        # constructor
        self.height = height
        self.width = width
        self.connections = connections
        self.goal_value = goal_values
        self.step_cost = step_cost
        self.discount_rate = discount_rate
        self.deadend_cost = -1000
    
    def initial_state_dist(self):
        # returns a distribution over initial states
        start_state = (1, self.height - 1)
        return DictDistribution({start_state: 1})

    def actions(self, s):
        # actions that are available at a state
        # aa = [s,]
        aa = []
        for s_, ns in self.connections:
            if s_ == s:
                aa.append(ns)
        if len(aa) == 0:
            aa.append(s)
        return aa

    def next_state_dist(self, s, a):
        # action a is a next state
        return DictDistribution({a: 1})

    def reward(self, s, a, ns):
        # return reward for state, action, next state
        reward = self.goal_value.get(ns, 0.0) 
        reward += self.step_cost
        if self.is_deadend(ns):
            reward += self.deadend_cost
        return reward
    
    def is_terminal(self, s):
        return self.is_goal(s) or self.is_deadend(s)
    
    def is_deadend(self, s):
        if self.is_goal(s):
            return False
        actions = self.actions(s)
        return len(actions) == 1 and s in actions
    
    def is_goal(self, s):
        return s[1] == 0
    
    def is_terminating(self):
        """Is a terminal state reachable?"""
        for s in self.reachable_states():
            if self.is_terminal(s):
                return True
        return False
    
    def has_solution(self):
        """Does this MDP have a solution (goal states reachable from starting state)?"""
        goals = [s for s in self.state_list if self.is_goal(s)]
        return any([goal in self.reachable_states() for goal in goals])

    def has_no_deadends(self):
        """Does this MDP have deadends (infinite self-loops)?"""
        for s in self.state_list:
            actions = self.actions(s)
            if len(actions) == 1 and s in actions and not self.is_goal(s):
                return False
        return True
    
    def plot(self,ax=None,plotgraph=True):
        gwp = GraphWorldPlotter(self, ax=ax)
        if plotgraph==True:
            gwp.plot()
        return gwp
    def plot_resp_dist(self,resp_counts,ax=None,plotgraph=True):
        gwp = GraphWorldPlotter(self, ax=ax)
        if plotgraph==True:
            # print(resp_counts)
            gwp.plot_resp_dist(resp_counts)
        return gwp
    
class GraphWorldPlotter(object):
    def __init__(self, mdp, ax=None):
        self.mdp = mdp
        if ax is None:
            _, ax = plt.subplots(1, 1)
        self.ax = ax
    
    def plot(self):
        all_states = sorted(self.mdp.goal_value.keys(), key=lambda s: (-s[1], s[0]))
        goalvalues = [self.mdp.goal_value[s] for s in all_states]
        self.ax.scatter(*zip(*all_states), c=goalvalues, s=500, edgecolors='k', zorder=3, cmap=myred)
        for si, s in enumerate(all_states):
            self.ax.text(s[0]+.08, s[1]-.2, str(si), va='center', ha='center', fontsize=14)
            reward = self.mdp.goal_value[s]
            reward = f"+{reward}" if reward > 0 else reward
            self.ax.text(
                s[0], s[1], reward, va='center', ha='center', fontsize=14
            )
        for s, ns in self.mdp.connections:
            self.ax.plot([s[0], ns[0]], [s[1], ns[1]], c="lightgrey", linestyle="dashed", zorder=0)
        self.ax.axis('off')
        
    def plot_resp_dist(self,resp_counts):
        all_states = sorted(self.mdp.goal_value.keys(), key=lambda s: (-s[1], s[0]))
        goalvalues = [self.mdp.goal_value[s] for s in all_states]
        self.ax.scatter(*zip(*all_states), c=goalvalues, s=500, edgecolors='k', zorder=3, cmap=myred)
        for si, s in enumerate(all_states):
            self.ax.text(s[0]+.08, s[1]-.2, str(si), va='center', ha='center', fontsize=14)
            reward = self.mdp.goal_value[s]
            reward = f"+{reward}" if reward > 0 else reward
            self.ax.text(
                s[0], s[1], reward, va='center', ha='center', fontsize=14
            )
        for s, ns in self.mdp.connections:
            self.ax.plot([s[0], ns[0]], [s[1], ns[1]], c="lightgrey", linestyle="dashed", zorder=0)
            
        resplist = list(resp_counts.keys())
        respdistcolor = self.respdistcolor(resp_counts)
        
        for resptype in resplist:
            # print(resp_counts[resptype])
            countcolor = respdistcolor[resp_counts[resptype],:]
            self.plot_advice(resptype,countcolor)
        

        self.plot_resp_colorscale(resp_counts)
        self.ax.axis('off')    
        
    def plot_trajectory(self, state_traj):
        for s, ns in zip(state_traj, state_traj[1:]): 
            self.ax.plot([s[0], ns[0]], [s[1], ns[1]], linewidth=5, c='k', zorder=1)
    
    def plot_advice(self, advice, color='blue'):
        s, ns = advice
        self.ax.plot([s[0], ns[0]], [s[1], ns[1]], linewidth=5, c=color, zorder=1)
    
    def plot_resp_colorscale(self,resp_counts):
        N = sum(list(resp_counts.values()))
        
        x    = np.linspace(1.5,2, N)
        y    = np.linspace(2.8,2.8, N)
        cols = np.linspace(0,1,N)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # lc = LineCollection(segments, cmap='Reds')
        # lc.set_array(cols)
        # lc.set_linewidth(10)
        # self.ax.add_collection(lc)
        # self.ax.text(x[0],y[0]-.15,"0", va='center', ha='center', fontsize=12)
        # self.ax.text(x[N-1],y[N-1]-.15,str(N), va='center', ha='center', fontsize=12)
        # self.ax.text(x[round(N/2)],y[N-1]+.15,"Count", va='center', ha='center', fontsize=12)
        
    def respdistcolor(self,resp_counts):
        N = sum(list(resp_counts.values()))
        return mpl.cm.Reds(np.linspace(0,1,N+1)) 
        
    def plot_advice_utility(self,advice_utility,yticklabels=True):
        # convet to simple keys from ((1,1),(0,0)) to (5,7)
        simple_keys=list(coordToint(advice_utility).keys())
        
        # # Normalize so sum of all advice is 1
        # if not sum(list(advice_utility.values()))==0:
        #     ut_norm=[i/sum(list(advice_utility.values())) for i in list(advice_utility.values())]
        # else:
        #     ut_norm = advice_utility.values()
        
        ut_norm = advice_utility.values()
        
        # Create a dataframe
        df = pd.DataFrame({'group':simple_keys, 'values':ut_norm })

        # Reorder it based on the values:     
        ordered_df = df.sort_values(by='group')    
        y_range=range(1,len(df.index)+1)
        
        if yticklabels==False:
            self.ax.hlines(y=y_range, xmax=0, xmin=-ordered_df['values'], color='#EF8330')
            self.ax.plot(-ordered_df['values'], y_range, "o",markeredgewidth=10,color='#262626')
            # self.ax.set_xlabel('normalized utility',size=12)
            self.ax.set_xlabel('expected utility',size=12)
            self.ax.yaxis.tick_right()
            self.ax.set_yticks(y_range, ordered_df['group'],size=15)
            self.ax.set_yticklabels([])

        else:
            self.ax.hlines(y=y_range, xmin=0, xmax=ordered_df['values'], color='#EF8330')
            self.ax.plot(ordered_df['values'], y_range, "o",markeredgewidth=10,color='#262626')
            self.ax.set_xlabel('normalized utility',size=12)
            self.ax.set_yticks(y_range, ordered_df['group'],size=15)

    def plot_advice_matching(self,advice_utility,yticklabels=True): 
        # Create a dataframe
        df = pd.DataFrame({'group':advice_utility.keys(), 'values':advice_utility.values() })

        # print(df)
        self.ax.hlines(y=df.index, xmin=0, xmax=df['values'], color='#262626',linewidth=30)
        # self.ax.plot(df['values'], df.index, "o",markeredgewidth=10)
        self.ax.set_xlabel('subjects',size=12)
        self.ax.set_yticks(df.index, df['group'],size=15)
        self.ax.set_ylim(-1, 2)

def coordToint(x):
    
    stateindex={
    (1,3):0,
    (0,2):1,
    (1,2):2,
    (2,2):3,
    (0,1):4,
    (1,1):5,
    (2,1):6,
    (0,0):7,
    (1,0):8,
    (2,0):9,
    }
    
    if type(x)== dict or issubclass(type(x), dict):
        dictionary = x
        keys = list(dictionary.keys())
        newkeys=[] 

        if type(list(dictionary.keys())[0][0]) is tuple:
            for i in range(len(keys)):
                from_node = stateindex[keys[i][0]]
                to_node = stateindex[keys[i][1]]
                newkeys.append([from_node,to_node])

            newkeys=tuple(map(tuple, newkeys))    

        elif type(list(dictionary.keys())[0][0]) is int:
            for i in range(len(keys)):
                newkeys.append(stateindex[keys[i]])

        return (dict(zip(newkeys,dictionary.values())))
    elif type(x)==tuple:
        Tuple = x
        return([stateindex[coord] for coord in Tuple])
    elif type(x) == frozenset:
        graph = x
        return [(stateindex[s0],stateindex[s1]) for s0,s1 in graph]
    else:
        print("please make sure input is tuple or dict")

def intTocoord(x):
    index={
        0:(1,3),
        1:(0,2),
        2:(1,2),
        3:(2,2),
        4:(0,1),
        5:(1,1),
        6:(2,1),
        7:(0,0),
        8:(1,0),
        9:(2,0)}
    if type(x)== dict:
        if type(list(x)[0])==  tuple:
            return {(index[s0] ,index[s1]):x[(s0,s1)] for s0,s1 in x}
        elif type(list(x)[0])==  int:
            return {(index[s]):x[(s)] for s in x}
        else:
            print('dick keys need to be tuple or int')
    elif type(x)==tuple:
        return(index[x])
    elif type(x)==set:
        return {(index[s0] ,index[s1]) for s0,s1 in x}
    elif type(x)==  int:
        return (index[x])
    else:
        print('need to be tuple, set, or dict')
    


    
    
    