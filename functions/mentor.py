"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

from abc import ABC, abstractmethod
import random
import itertools
from frozendict import frozendict

from msdm.core.distributions import DictDistribution
from msdm.algorithms import ValueIteration
from functions.functions import calc_teacher_graph_prior, solve_mdp, create_mdp
from functions.subgraphs import all_subgraph_solutions, clear_subgraph_cache
from functions.features import calc_obm_teacher_graph_prior,calc_heuristic_teacher_graph_prior, clear_graph_score

class BaseMentor(object):
    def __init__(
        self,
        true_mdp_params,
        teacher_mdp_params = None,
    ):
        self.true_mdp_params = true_mdp_params
        self.teacher_mdp_params = teacher_mdp_params if teacher_mdp_params is not None else true_mdp_params
        
    @abstractmethod
    def advice_dist(self, mentee_trajectories):
        pass
    
    @abstractmethod
    def advice_dist_sample(self, mentee_trajectories, n_samples):
        pass
    
    def advice_gain(self, advice, current_graph):
        """Gain in learner's value on task for a piece of advice given the learner has a graph"""
        
        current_learner_mdp_params = {**self.true_mdp_params, 'connections': current_graph}
        current_value = all_subgraph_solutions(frozendict(self.true_mdp_params))[frozendict(current_learner_mdp_params)].initial_value        

        new_graph = current_graph | frozenset([advice])
        new_learner_mdp_params = {**self.true_mdp_params, 'connections': new_graph}
        new_value = all_subgraph_solutions(frozendict(self.true_mdp_params))[frozendict(new_learner_mdp_params)].initial_value
      
        return new_value - current_value
    
    def advice_expected_utility(self, graph_dist):
        advice_exp_utility = {}
        # true_mdp = create_mdp(frozendict(self.true_mdp_params))
        teacher_mdp = create_mdp(frozendict(self.teacher_mdp_params))
        for advice in teacher_mdp.connections:
            advice_exp_utility[advice] = 0
            for current_graph, prob in graph_dist.items():
                advice_exp_utility[advice] += self.advice_gain(advice, current_graph)*prob
        return advice_exp_utility
    
class OptimalBayesianMentor(BaseMentor):

    def mentee_graph_posterior(self, mentee_trajectories):
        """The mentor's beliefs about what the learner's graph is"""
        assert len(mentee_trajectories) == 1, "Only handles a single traj"
        
        # graph_prior = calc_teacher_graph_prior(frozendict(self.true_mdp_params))
        
        result_temp = all_subgraph_solutions(frozendict(self.true_mdp_params))
        graph_prior = calc_obm_teacher_graph_prior(result_temp)
        # P_T(w \mid \zeta)
        mentee_traj = mentee_trajectories[0]
        
        def trajectoryProb(world):
            tot_temp_params = {**self.true_mdp_params, "connections": world}
            result_temp = all_subgraph_solutions(frozendict(self.true_mdp_params))[frozendict(tot_temp_params)]
            traj_prob = 1.0
            for s, ns in zip(mentee_traj, mentee_traj[1:]):
                action_prob = result_temp.policy.action_dist(s).prob(ns)
                traj_prob *= action_prob
                if traj_prob == 0.0:
                    return 0.0
            return traj_prob
        graph_posterior = graph_prior.factor(trajectoryProb)

        return graph_posterior
        
    def advice_dist(self, mentee_trajectories):
        graph_posterior = self.mentee_graph_posterior(mentee_trajectories)
        advice_exp_utility = self.advice_expected_utility(graph_posterior)
        return advice_exp_utility

    def advice_dist_sample(self, mentee_trajectories,n_samples):
        graph_posterior = self.mentee_graph_posterior(mentee_trajectories)
        sampled_graph_posterior = dict(random.sample(list(graph_posterior.items()), n_samples))
        advice_exp_utility = self.advice_expected_utility(sampled_graph_posterior)
        return advice_exp_utility

class NaiveBayesianMentor(BaseMentor):

    def mentee_graph_posterior(self, mentee_trajectories):
        """The mentor's beliefs about what the learner's graph is"""
        assert len(mentee_trajectories) == 1, "Only handles a single traj"
        
        # graph_prior = calc_teacher_graph_prior(frozendict(self.true_mdp_params))
        result_temp = all_subgraph_solutions(frozendict(self.true_mdp_params))
        graph_prior = calc_obm_teacher_graph_prior(result_temp)
        
        # P_T(w \mid \zeta)
        mentee_traj = mentee_trajectories[0]
        
        def trajectoryProb(world):
            return set([(s,ns) for s,ns in zip(mentee_traj[:-1],mentee_traj[1:])]).issubset(world)
        
        graph_posterior = graph_prior.factor(trajectoryProb)

        return graph_posterior
        
    def advice_dist(self, mentee_trajectories):
        graph_posterior = self.mentee_graph_posterior(mentee_trajectories)
        advice_exp_utility = self.advice_expected_utility(graph_posterior)
        return advice_exp_utility
    
class PriorOnlyMentor(BaseMentor):
    def mentee_graph_prior(self):
        """The mentor's beliefs about what the learner's graph is"""
        graph_prior = calc_teacher_graph_prior(frozendict(self.true_mdp_params))
        return graph_prior
    
    def prior_advice_dist(self):
        graph_posterior = self.mentee_graph_prior()
        advice_exp_utility = self.advice_expected_utility(graph_posterior)
        return advice_exp_utility
