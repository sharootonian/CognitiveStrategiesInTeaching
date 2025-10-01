"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.special import logsumexp,expit
from scipy.optimize import minimize

def get_policy(advice_utility, invtemp):
    vals = np.array(list(advice_utility.values()))
    keys = list(advice_utility.keys())
    logits = invtemp * (vals - np.max(vals))  # subtract max for stability
    exps = np.exp(logits - logsumexp(logits))  # normalized, safe
    return dict(zip(keys, exps))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def create_nll_features(trials, feature_names, choice_col, fit_randchoose=True):    
    def nll_features(feature_weights):
        if fit_randchoose:
            assert len(feature_names) == len(feature_weights) - 1
            # randchoose = sigmoid(feature_weights[-1])
            randchoose = expit(feature_weights[-1])
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
            eps = 1e-12
            total_logprob += np.log(probs[edge] + eps)
        return -total_logprob
    return nll_features

def fitting_choices(preprocessed, models,exp):

    model_fits = []
    subj_list = preprocessed['subjID'].unique().tolist()
    n = 40
    
    for subj in tqdm(subj_list):
        trials = preprocessed[preprocessed['subjID']== subj]
        for model_name, feature_names in models.items():
            feature_weights = [0]*(len(feature_names))
            nll_func = create_nll_features(trials, feature_names, "edge", fit_randchoose=False)
            min_res = minimize(nll_func,feature_weights,method = 'SLSQP')
            res = dict(zip(feature_names + ["randchoose_logit"], min_res.x))
            # res['randchoose'] = sigmoid(res['randchoose_logit'])
            res['model'] = model_name
            res['nparam'] = len(feature_names)
            res['subjID'] = subj
            res['min_success'] = min_res.success
            res['min_res'] = min_res
            res['nll'] = min_res.fun
            
            if exp >=2:
                res['group'] = list(trials.group)[0]
                n = 5
                if exp == 3:
                    n = 5
                    res['condition'] = list(trials.condition)[0]
                    res['subjectId'] = list(trials.subjectId)[0]
                    
            model_fits.append(res)
            
    df_fits= pd.DataFrame(model_fits)
    for i, row in df_fits.iterrows():
        df_fits.at[i,'BIC'] = np.log(n)*row.nparam + 2*row.nll
        
    return df_fits

def model_recovery_fitting(posterior_sim, models):
    model_fits = []
    subj_list = posterior_sim['subjID'].unique().tolist()
    samples = posterior_sim['iteration'].unique().tolist()
    for s in  tqdm(samples,desc='iterations'):
        posterior_sim_s= posterior_sim[posterior_sim['iteration']== s]
        for subj in subj_list:
            trials = posterior_sim_s[posterior_sim_s['subjID']== subj]
            for sim_model in trials.sim_model.unique():
                trial = trials[trials.sim_model == sim_model].reset_index(drop=True)
                for model_name, feature_names in models.items():
                    feature_weights = [0]*(len(feature_names))
                    nll_func = create_nll_features(trial, feature_names, "choice", fit_randchoose=False)
                    min_res = minimize(nll_func,feature_weights,method = 'SLSQP')
                    betas = list(np.atleast_1d(min_res.x))
                    res = {
                        "sample": s,
                        "subjID": subj,
                        "sim_model": sim_model,     # the model that generated the data (if sim)
                        "fit_model": model_name,      # the model used to fit
                        "nparam": len(betas),
                        "nll": float(min_res.fun),
                        "BIC": np.log(5)*len(betas) + 2*float(min_res.fun)
                    }
                    res.update({f"beta{i+1}": b for i, b in enumerate(betas)})                     
                    model_fits.append(res)
        df_fits= pd.DataFrame(model_fits)
    return df_fits