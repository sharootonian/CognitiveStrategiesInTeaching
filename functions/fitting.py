import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import minimize_scalar

def get_policy(advice_utility, invtemp):
    max_advice_utility = max(advice_utility.values())
    advice_utility = {e: u - max_advice_utility for e, u in advice_utility.items()}
    probs = {edge: np.exp(invtemp*utility) for edge, utility in advice_utility.items()}
    norm = sum(probs.values())
    probs = {e: p/norm for e, p in probs.items()}
    return probs

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def create_nll_features(trials, feature_names, choice_col, fit_randchoose=True):    
    def nll_features(feature_weights):
        if fit_randchoose:
            assert len(feature_names) == len(feature_weights) - 1
            randchoose = sigmoid(feature_weights[-1])
            feature_weights = feature_weights[:-1]
        else:
            assert len(feature_names) == len(feature_weights)
            randchoose = 0
        total_logprob = 0
        for _, trial in trials.iterrows():
            edge_logits = defaultdict(float)
            for weight, feature_name, edge_featurevalue_dict in zip(feature_weights, feature_names, trial[feature_names]):
                for edge, val in edge_featurevalue_dict.items():
                    edge_logits[edge] += val*weight
                    
            probs = get_policy(edge_logits, 1)
            probs = {e: (1 - randchoose)*p + randchoose/len(probs) for e, p in probs.items()}
            edge = trial[choice_col]
            total_logprob += np.log(probs[edge])
        return -total_logprob
    return nll_features
