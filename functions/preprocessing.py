"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import numpy as np
import pandas as pd
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from os.path import dirname, join
from frozendict import frozendict
from functions.graphWorld import GraphWorld,coordToint, intTocoord
from functions.functions import max_value_keys,resptograph_count,matching_count,solve_mdp,create_mdp
from functions.mdp_params import make_true_graph,create_random_mdp_params,create_true_mdp_params
from functions.features import get_feature_levels, get_feature_reward_sum_rmTraj,get_feature_reward_sum


stringtoset = lambda x_to_y : tuple([int(i) for i in x_to_y.split('to')])

flipper={
        (0,1):(0,3),
        (0,2):(0,2),
        (0,3):(0,1),
        (1,4):(3,6),
        (1,5):(3,5),
        (2,4):(2,6),
        (2,5):(2,5),
        (2,6):(2,4),
        (3,5):(1,5),
        (3,6):(1,4),
        (4,7):(6,9),
        (4,8):(6,8),
        (5,7):(5,9),
        (5,8):(5,8),
        (5,9):(5,7),
        (6,8):(4,8),
        (6,9):(4,7),
        }

true_graph = make_true_graph(4)

def preprocessing(trialSetup,simdata,RAW_DIR):

    ## Locate files.
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('json')])

    ## initialize df
    fulldata = pd.DataFrame()
    data = pd.DataFrame()
    preprocessed = pd.DataFrame() 
    preprocessed_subj = pd.DataFrame({'subjectId':0,
                                 'subjID':0,
                                 'index':0,
                                 'simID':0,
                                 'trial':0,
                                 'trialtype':0,     
                                 'seed':0,
                                 'flipped':0,
                                 'rt':0,
                                 'edgeClickLog':[0],
                                 'edgeClickNum':0,
                                 'edgeRT':0,
                                 'edgeRT_zscore':0,
                                 'edgeRT_log':0,
                                 'edgeRT_log_zscore':0,
                                 'edgeRT_norm':0,
                                 'edgeRT_log_norm':0,
                                 'edge':[(0,0)],
                                 'OBM_AU':[(0,0)],
                                 'NBM_AU':[(0,0)],
                                 'POM_AU':[(0,0)],
                                 'OBM_maxUA':[(0,0)],
                                 'NBM_maxUA':[(0,0)],
                                 'POM_maxUA':[(0,0)], 
                                 'traj':[(0,0)],
                                 'feature_levels':[(0,0)],
                                 'feature_reward_sum':[(0,0)], 
                                 'feature_reward_max':[(0,0)],
                                 'feature_distance_FW':[(0,0)],
                                 'feature_opt_traj_edge':[(0,0)],
                                 'learner_rewards':[(0,0)],
                                 'PathAvgUtility':[(0,0)],
                                 'Q-values':[(0,0)],                                
                                }
                               )
    bonus=dict()
    task_dur=dict()

    debrief_q = ['How mentally demanding was the task?',
               'How clear were the task instructions?',
               'How successful were you in accomplishing what you were asked to do during the task?',
               'How hard did you have to work to accomplish your level of performance?',
               'How discouraged, irritated, stressed, or annoyed were you during the task?',
               "Did you use any strategies during the task? <br> (e.g. write things down)",
               "Do you have any other comments or feedback?",
                ]

    debrief = {'subjID':[],
               'P0_Q1':[],
               'P0_Q2':[],
               'P0_Q3':[],
               'P0_Q4':[],
               'P0_Q5':[],
               'strategies':[],
               'feedback':[],
              }

    #loop through each subject and append, process, compute bonus and debrief 
    for i in range(0,len(files)):
        # get file name
        f=files[i]

        # read json file into pd df
        fulldata_subj = pd.read_json(os.path.join(RAW_DIR, f))

        ## Define subject
        fulldata_subj['subjectId']=f.replace('.json','')
        fulldata_subj['subjID']=subj="s"+str(i)

        # index trial data
        if len(fulldata_subj)==50:
            data_subj = fulldata_subj.iloc[8:48].reset_index(drop=True)
            debrief_subj = fulldata_subj.loc[48,'response']
        elif len(fulldata_subj)==51:
            data_subj = fulldata_subj.iloc[9:49].reset_index(drop=True)
            debrief_subj = fulldata_subj.loc[49,'response']
        else:
            print("data_subject is a diffrent size")

        debrief_subj['subjID']=fulldata_subj["subjID"][0]

        # append debriefing responses
        for key in list(debrief.keys()):
            debrief[key].append(debrief_subj[key])


        for i in data_subj.index:
            preprocessed_subj.loc[i,'subjectId'] = data_subj.loc[i,'subjectId']
            preprocessed_subj.loc[i,'subjID'] = data_subj.loc[i,'subjID']
            preprocessed_subj.loc[i,'index'] = data_subj.loc[i,'index']
            preprocessed_subj.loc[i,'simID'] = trialSetup[trialSetup['index']== preprocessed_subj.loc[i,'index']]['sim_num'].tolist()[0]
            preprocessed_subj.loc[i,'trial'] = i
            preprocessed_subj.loc[i,'trialtype'] = str(int(data_subj.loc[i,'flipped']=='TRUE'))+'_'+str(int(data_subj.loc[i,'seed']))
            preprocessed_subj.loc[i,'seed'] = data_subj.loc[i,'seed']
            preprocessed_subj.at[i,'flipped'] = data_subj.loc[i,'flipped']
            preprocessed_subj.loc[i,'edgeRT'] = data_subj.loc[i,'edgeRT']
            preprocessed_subj.loc[i,'edgeRT_zscore'] = (np.float64(data_subj.loc[i, 'edgeRT']) - (data_subj['edgeRT'].astype(float).mean())) / (data_subj['edgeRT'].astype(float).std())
            preprocessed_subj.loc[i,'edgeRT_log'] = np.log(np.float64(data_subj.loc[i,'edgeRT']))
            preprocessed_subj.loc[i,'edgeRT_log_zscore'] = (np.log(np.float64(data_subj.loc[i,'edgeRT'])) - np.log((data_subj['edgeRT'].astype(float)).mean())) / np.log(data_subj['edgeRT'].astype(float)).std()
            preprocessed_subj.loc[i,'rt'] = data_subj.loc[i,'rt']
            preprocessed_subj.loc[i,'edgeClickLog'] = data_subj.loc[i,'edgeClickLog']
            preprocessed_subj.loc[i,'edgeClickNum'] = len(eval(data_subj.loc[i,'edgeClickLog']))
            preprocessed_subj.at[i,'edge'] = stringtoset(data_subj.loc[i,'edge'])
            if(data_subj.loc[i,'flipped']=='TRUE'):# flip back the flipped trials
                preprocessed_subj.at[i,'edge'] = flipper[stringtoset(data_subj.loc[i,'edge'])]
        
        meanRT = np.sum(preprocessed_subj.edgeRT)
        meanRT_log = np.sum(preprocessed_subj.edgeRT_log)

        for i in preprocessed_subj.index:
            #match by seed
            trial = simdata.loc[simdata['seed'] == preprocessed_subj['seed'][i]]
            preprocessed_subj.loc[i,'edgeRT_norm'] = preprocessed_subj.loc[i,'edgeRT']/meanRT
            preprocessed_subj.loc[i,'edgeRT_log_norm'] = preprocessed_subj.loc[i,'edgeRT_log']/meanRT
            seed = (preprocessed_subj['seed'][i])
            
            #get OBM_advice_utility and convert ti simple nodes
            preprocessed_subj.at[i,'OBM_AU'] = coordToint(trial['OBM_advice_utility'].tolist()[0])
            preprocessed_subj.loc[i,'edge_utility_OBM_AU'] =  preprocessed_subj.loc[i,'OBM_AU'][preprocessed_subj.loc[i,'edge']]
            
            preprocessed_subj.at[i,'NBM_AU'] = coordToint(trial['NBM_advice_utility'].tolist()[0])
            preprocessed_subj.loc[i,'NBM_maxUA'] =  preprocessed_subj.loc[i,'NBM_AU'][preprocessed_subj.loc[i,'edge']]

            #quick dirty way to find max bonus pay
            # preprocessed_subj.loc[i,'edge_utility'] =  preprocessed_subj.loc[i,'OBM_AU'][0][max_value_keys(preprocessed_subj.loc[i,'OBM_AU'][0])[0]]

            #get POM_advice_utility and convert ti simple nodes
            preprocessed_subj.at[i,'POM_AU'] = coordToint(trial['POM_advice_utility'].tolist()[0])
            preprocessed_subj.loc[i,'edge_utility_POM_AU_'] =  preprocessed_subj.loc[i,'POM_AU'][preprocessed_subj.loc[i,'edge']]

            #get best advice given each model
            preprocessed_subj.at[i,'OBM_maxUA'] = max_value_keys(preprocessed_subj.loc[i,'OBM_AU'])[0]
            preprocessed_subj.at[i,'POM_maxUA'] = max_value_keys(preprocessed_subj.loc[i,'POM_AU'])[0]
            
            preprocessed_subj.at[i,'PathAvgUtility'] = coordToint(trial['PathAvgUtility'].tolist()[0]) 
            preprocessed_subj.at[i,'Q-values'] = coordToint(trial['Q-values'].tolist()[0]) 
            
            preprocessed_subj.loc[i,'edge_max_U_obm'] =  preprocessed_subj.loc[i,'OBM_AU'][preprocessed_subj.at[i,'OBM_maxUA']]
            preprocessed_subj.at[i,'edge_normresp_U_obm'] =  np.divide(preprocessed_subj.loc[i,'edge_utility_OBM_AU'],preprocessed_subj.loc[i,'edge_max_U_obm'])
            
            preprocessed_subj.loc[i,'learner_rewards'] = [trial.learner_rewards.tolist()[0]]
            
            rewards = trial.learner_rewards.tolist()[0]
            traj = trial.traj.tolist()[0]
            
            preprocessed_subj.at[i,'traj'] = traj
            
            # get features
            preprocessed_subj.at[i,'feature_levels'] = coordToint(get_feature_levels(true_graph))
            preprocessed_subj.at[i,'feature_reward_sum'] = coordToint(get_feature_reward_sum(rewards,true_graph))

            
            
        #compute bonus pay
        bonus[preprocessed_subj['subjID'].unique()[0]]= sum(preprocessed_subj['edge_utility_OBM_AU'])*(6/5.33)
        # bonus[preprocessed_subj['subjID'].unique()[0]]= sum(preprocessed_subj['edge_utility_true'])

        #total time
        task_dur[fulldata_subj['subjID'].unique()[0]] = fulldata_subj.iloc[-1]['time_elapsed']/60000

        # append to one big df
        fulldata = pd.concat([fulldata,fulldata_subj])
        data = pd.concat([fulldata,fulldata_subj])
        preprocessed = pd.concat([preprocessed,preprocessed_subj])
        preprocessed = preprocessed.reset_index(drop=True)
        
    # preprocessed = remove_bad_subjects(remove_invalid_trials(preprocessed))
        
    return preprocessed,bonus,task_dur, debrief_q, debrief

def preprocessing_exp2(simdata,RAW_DIR):
    ## Locate files.
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('json')])

    ## initialize df
    fulldata = pd.DataFrame()
    data = pd.DataFrame()
    preprocessed = pd.DataFrame() 
    preprocessed_subj = pd.DataFrame({'subjectId':0,
                                 'subjID':0,
                                 'index':0,
                                 'trial':0,
                                 'trialtype':0,
                                 'trial_id':0,
                                 'congruency':0,
                                 'flipped':0,
                                 'rt':0,
                                 'skip':[0],
                                 'block':[0],
                                 'group':[0],
                                 'edgeClickLog':[0],
                                 'edgeClickNum':0,
                                 'edgeRT':0,
                                 'edgeRT_zscore':0,
                                 'edgeRT_log':0,
                                 'edgeRT_log_zscore':0,
                                 'edgeRT_norm':0,
                                 'edgeRT_log_norm':0,
                                 'edge':[(0,0)],
                                 'U_obm':[(0,0)],
                                 'U_obm_max':[(0,0)],
                                 'U_nbm':[(0,0)],
                                 'U_fr':[(0,0)],
                                 'U_rand':[(0,0)],
                                 'U_obm_max_edge': [(0,0)],
                                 'traj':[(0,0)],
                                 'feature_levels':[(0,0)],
                                 'feature_reward_sum':[(0,0)], 
                                 # 'feature_opt_traj_edge':[(0,0)],
                                }
                               )
    bonus=dict()
    task_dur=dict()

    debrief_q = ['How mentally demanding was the task?',
               'How clear were the task instructions?',
               'How successful were you in accomplishing what you were asked to do during the task?',
               'How hard did you have to work to accomplish your level of performance?',
               'How discouraged, irritated, stressed, or annoyed were you during the task?',
               "Did you use any strategies during the task? <br> (e.g. write things down)",
               "Do you have any other comments or feedback?",
                ]

    debrief = {'subjID':[],
               'P0_Q1':[],
               'P0_Q2':[],
               'P0_Q3':[],
               'P0_Q4':[],
               'P0_Q5':[],
               'strategies':[],
               'feedback':[],
              }
    for i in range(0,len(files)):
        f=files[i]

        # read json file into pd df
        fulldata_subj = pd.read_json(os.path.join(RAW_DIR, f))

        ## Define subject
        fulldata_subj['subjectId']=f.replace('.json','')
        fulldata_subj['subjID']=subj="s"+str(i)

        # index trial data
        data_subj = fulldata_subj[fulldata_subj.trial_type =='external-html'].reset_index(drop=True)
        debrief_subj = fulldata_subj[fulldata_subj.trial_type =='survey'].reset_index(drop=True).loc[0,'response']
        if (filter_skipped_trials(data_subj)[0] <= 0.50):  
            continue
        else:
            data_subj = filter_skipped_trials(data_subj)[1] 
            
            
        debrief_subj['subjID']=fulldata_subj["subjID"][0]

        # append debriefing responses
        for key in list(debrief.keys()):
            debrief[key].append(debrief_subj[key])

        for i in data_subj.index:
            preprocessed_subj.loc[i,'subjectId'] = data_subj.loc[i,'subjectId']
            preprocessed_subj.loc[i,'subjID'] = data_subj.loc[i,'subjID']
            preprocessed_subj.loc[i,'index'] = data_subj.loc[i,'index']
            preprocessed_subj.loc[i,'trial'] = i
            preprocessed_subj.loc[i,'congruency'] = data_subj.congruency[i]
            preprocessed_subj.loc[i,'trial_id'] = data_subj.trial_id[i]
            preprocessed_subj.loc[i,'block'] = "test" if data_subj.trial_id[i].startswith('test') else "training" 
            # preprocessed_subj.loc[i,'seed'] = data_subj.loc[i,'seed']
            preprocessed_subj.at[i,'flipped'] = data_subj.loc[i,'flipped']
            preprocessed_subj.loc[i,'edgeRT'] = np.float64(data_subj.loc[i,'edgeRT'])
            preprocessed_subj.loc[i,'edgeRT_zscore'] = (np.float64(data_subj.loc[i, 'edgeRT']) - (data_subj['edgeRT'].astype(float).mean())) / (data_subj['edgeRT'].astype(float).std())
            preprocessed_subj.loc[i,'edgeRT_log'] = np.log(np.float64(data_subj.loc[i,'edgeRT']))
            preprocessed_subj.loc[i,'edgeRT_log_zscore'] = (np.log(np.float64(data_subj.loc[i,'edgeRT'])) - np.log((data_subj['edgeRT'].astype(float)).mean())) / np.log(data_subj['edgeRT'].astype(float)).std()
            preprocessed_subj.loc[i,'rt'] = data_subj.loc[i,'rt']
            preprocessed_subj.loc[i,'skip'] = data_subj.loc[i,'skip']
            preprocessed_subj.loc[i,'edgeClickLog'] = data_subj.loc[i,'edgeClickLog']
            preprocessed_subj.loc[i,'edgeClickNum'] = len(eval(data_subj.loc[i,'edgeClickLog']))
            preprocessed_subj.at[i,'edge'] = stringtoset(data_subj.loc[i,'edge'])
            if((data_subj.loc[i,'flipped']=='True') | (data_subj.loc[i,'flipped']=='TRUE')):# flip back the flipped trials
                preprocessed_subj.at[i,'edge'] = flipper[stringtoset(data_subj.loc[i,'edge'])]
            if data_subj.loc[1,'congruency']=='C':
                preprocessed_subj.at[i,'group'] = 'CI'
            else:
                preprocessed_subj.at[i,'group'] = 'II'
        
#       trial type collapses across flipped so we can look at avg across both
        preprocessed_subj['trialtype'] = [i[:8] if(i[0]=='i') else i[:6] for i in preprocessed_subj.trial_id]
        
        preprocessed_subj.edgeRT = pd.to_numeric(preprocessed_subj.edgeRT)
        
        # clear out repeated trial values from some exp bug
        preprocessed_subj = preprocessed_subj.drop_duplicates(subset='trial_id').reset_index(drop=True).copy()

        meanRT = np.sum(preprocessed_subj.edgeRT)
        meanRT_log = np.sum(preprocessed_subj.edgeRT_log)
        
        for i in preprocessed_subj.index:
            # match by trial type
            trial = simdata.loc[simdata.trial_id == preprocessed_subj['trialtype'][i]]
            preprocessed_subj.loc[i,'edgeRT_norm'] = preprocessed_subj.loc[i,'edgeRT']/meanRT
            preprocessed_subj.loc[i,'edgeRT_log_norm'] = preprocessed_subj.loc[i,'edgeRT_log']/meanRT_log
            
            #get model utility for resp    
            preprocessed_subj.at[i,'U_obm'] = trial['U_obm'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_obm'] =  preprocessed_subj.loc[i,'U_obm'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_obm_max'] = trial['U_obm_max'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_obm_max'] =  preprocessed_subj.loc[i,'U_obm_max'][preprocessed_subj.loc[i,'edge']]
            
            preprocessed_subj.at[i,'U_nbm'] = trial['U_nbm'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_nbm'] =  preprocessed_subj.loc[i,'U_nbm'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_fr'] = trial['U_fr'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_fr'] =  preprocessed_subj.loc[i,'U_fr'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_rand'] = trial['U_rand'].tolist()[0]
            preprocessed_subj.loc[i,'edge_utility_U_rand'] =  preprocessed_subj.loc[i,'U_rand'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_obm_max_edge'] = max_value_keys(preprocessed_subj.loc[i,'U_obm'])[0]
            
            #get max model ulitliy
            preprocessed_subj.loc[i,'edge_max_U_obm'] =  preprocessed_subj.loc[i,'U_obm'][max_value_keys(preprocessed_subj.loc[i,'U_obm'])[0]]
            preprocessed_subj.loc[i,'edge_max_U_nbm'] =  preprocessed_subj.loc[i,'U_nbm'][max_value_keys(preprocessed_subj.loc[i,'U_nbm'])[0]]
            preprocessed_subj.loc[i,'edge_max_U_fr'] =  preprocessed_subj.loc[i,'U_fr'][max_value_keys(preprocessed_subj.loc[i,'U_fr'])[0]]
            preprocessed_subj.loc[i,'edge_max_U_fr_bonus'] =  preprocessed_subj.loc[i,'U_obm'][max_value_keys(preprocessed_subj.loc[i,'U_fr'])[0]]
            preprocessed_subj.loc[i,'edge_max_U_rand'] =  preprocessed_subj.loc[i,'U_rand'][max_value_keys(preprocessed_subj.loc[i,'U_rand'])[0]]
            
            preprocessed_subj.loc[i,'edge_normresp_U_obm'] =  np.divide(preprocessed_subj.loc[i,'edge_resp_U_obm'],preprocessed_subj.loc[i,'edge_max_U_obm'])
            preprocessed_subj.loc[i,'edge_normresp_U_fr_bonus'] =  np.divide(preprocessed_subj.loc[i,'edge_max_U_fr_bonus'],preprocessed_subj.loc[i,'edge_max_U_obm'])
            preprocessed_subj.loc[i,'edge_normresp_U_fr'] =  np.divide(preprocessed_subj.loc[i,'edge_resp_U_fr'],preprocessed_subj.loc[i,'edge_max_U_fr'])

        #     #quick dirty way to find max bonus pay
            preprocessed_subj.loc[i,'maxbonuscalc'] =  preprocessed_subj.loc[i,'U_obm'][max_value_keys(preprocessed_subj.loc[i,'U_obm'])[0]]

            rewards = intTocoord(trial.goal_values.tolist()[0])
            # print(trial.connections.tolist()[0])
            traj = get_traj_from_connections(set(intTocoord(trial.connections.tolist()[0])))

            preprocessed_subj.at[i,'traj'] = traj

            # get features
            preprocessed_subj.at[i,'feature_levels'] = coordToint(get_feature_levels(true_graph))
            preprocessed_subj.at[i,'feature_reward_sum'] = coordToint(get_feature_reward_sum_rmTraj(rewards,true_graph,traj))

            
        #compute bonus pay
        bonus[preprocessed_subj['subjID'].unique()[0]]= sum(preprocessed_subj['edge_resp_U_obm'])
        # bonus[preprocessed_subj['subjID'].unique()[0]]= sum(preprocessed_subj['edge_utility_true'])

        #total time
        task_dur[fulldata_subj['subjID'].unique()[0]] = fulldata_subj.iloc[-1]['time_elapsed']/60000

        # append to one big df
        fulldata = pd.concat([fulldata,fulldata_subj])
        data = pd.concat([fulldata,fulldata_subj])
        preprocessed = pd.concat([preprocessed,preprocessed_subj])
        preprocessed = preprocessed.reset_index(drop=True)

    return preprocessed,bonus,task_dur, debrief_q, debrief

def preprocessing_exp3(simdata,RAW_DIR):
    ## Locate files.
    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('json')])

    ## initialize df
    fulldata = pd.DataFrame()
    data = pd.DataFrame()
    preprocessed = pd.DataFrame() 
    preprocessed_subj = pd.DataFrame({'subjectId':0,
                                    'subjID':0,
                                    'index':0,
                                    'trial':0,
                                    'trialtype':0,
                                    'trial_id':0,
                                    'trial_group':0,
                                    'congruency':0,
                                    'flipped':0,
                                    'rt':0,
                                    'skip':[0],
                                    'block':[0],
                                    'group':[0],
                                    'edgeClickLog':[0],
                                    'edgeClickNum':0,
                                    'edgeRT':0,
                                    'edgeRT_zscore':0,
                                    'edgeRT_log':0,
                                    'edgeRT_log_zscore':0,
                                    'edgeRT_norm':0,
                                    'edgeRT_log_norm':0,
                                    'edge':[(0,0)],
                                    'inference_edge':[(0,0)],
                                    'inference_edgeClickLog':0,
                                    'inference_edgeClickNum':0,
                                    'inference_edgeRT':0,
                                    'inference_edgeRT_zscore':0,
                                    'inference_edgeRT_log':0,
                                    'inference_edgeRT_log_zscore':0,
                                    'inference_edge_last':[(0,0)],
                                    'condition':0,
                                    'inf_prob':[(0,0)],
                                    'inf_prob_max':[(0,0)],
                                    'inf_prob_resp':[(0,0)],
                                    'inf_prob_norm':[(0,0)],
                                    'U_obm':[(0,0)],
                                    'U_obm_max':[(0,0)],
                                    'U_nbm':[(0,0)],
                                    'U_fr':[(0,0)],
                                    'U_rand':[(0,0)],
                                    'U_obm_max_edge': [(0,0)],
                                    'traj':[(0,0)],
                                    'feature_levels':[(0,0)],
                                    'feature_reward_sum':[(0,0)], 
                                    # 'feature_opt_traj_edge':[(0,0)],
                                }
                                )
    bonus=dict()
    task_dur=dict()

    debrief_q = ['How mentally demanding was the task?',
                'How clear were the task instructions?',
                'How successful were you in accomplishing what you were asked to do during the task?',
                'How hard did you have to work to accomplish your level of performance?',
                'How discouraged, irritated, stressed, or annoyed were you during the task?',
                "Did you use any strategies during the task? <br> (e.g. write things down)",
                "Do you have any other comments or feedback?",
                ]

    debrief = {'subjID':[],
                'P0_Q1':[],
                'P0_Q2':[],
                'P0_Q3':[],
                'P0_Q4':[],
                'P0_Q5':[],
                'strategies':[],
                'feedback':[],
                }
    for i in range(0,len(files)):
        f=files[i]

        # read json file into pd df
        fulldata_subj = pd.read_json(os.path.join(RAW_DIR, f))

        ## Define subject
        fulldata_subj['subjectId']=f.replace('.json','')
        fulldata_subj['subjID']=subj="s"+str(i)

        # index trial data
        data_subj = fulldata_subj[fulldata_subj.trial_type =='external-html'].reset_index(drop=True)
        debrief_subj = fulldata_subj[fulldata_subj.trial_type =='survey'].reset_index(drop=True).loc[0,'response']
        
        if (filter_skipped_trials(data_subj)[0] <= 0.50):  
            continue
        else:
            data_subj = filter_skipped_trials(data_subj)[1] 

            

        debrief_subj['subjID']=fulldata_subj["subjID"][0]

        # append debriefing responses
        for key in list(debrief.keys()):
            debrief[key].append(debrief_subj[key])

        for i in data_subj.index:
            preprocessed_subj.loc[i,'subjectId'] = data_subj.loc[i,'subjectId']
            preprocessed_subj.loc[i,'subjID'] = data_subj.loc[i,'subjID']
            preprocessed_subj.loc[i,'index'] = data_subj.loc[i,'index']
            preprocessed_subj.loc[i,'trial'] = i
            preprocessed_subj.loc[i,'congruency'] = data_subj.congruency[i]
            preprocessed_subj.loc[i,'condition'] = data_subj.condition[i]
            preprocessed_subj.loc[i,'trial_id'] = data_subj.trial_id[i]
            
            if data_subj.trial_group[i] == 0:
                preprocessed_subj.loc[i,'trial_group'] = '5to10'
            elif data_subj.trial_group[i] == 1:
                preprocessed_subj.loc[i,'trial_group'] = '1to5'
            elif data_subj.trial_group[i] == 2:
                preprocessed_subj.loc[i,'trial_group'] = '10to15'
            elif data_subj.trial_group[i] == 3:
                preprocessed_subj.loc[i,'trial_group'] = '15to20'
            else:         
                preprocessed_subj.loc[i,'trial_group'] = data_subj.trial_group[i]
            preprocessed_subj.loc[i,'block'] = "test" if data_subj.trial_id[i].startswith('test') else "training" 
            # preprocessed_subj.loc[i,'seed'] = data_subj.loc[i,'seed']
            preprocessed_subj.at[i,'flipped'] = data_subj.loc[i,'flipped']
            preprocessed_subj.loc[i,'edgeRT'] = np.float64(data_subj.loc[i,'edgeRT'])
            
            preprocessed_subj.loc[i,'edgeRT_zscore'] = (np.float64(data_subj.loc[i, 'edgeRT']) - (data_subj['edgeRT'].astype(float).mean())) / (data_subj['edgeRT'].astype(float).std())
            preprocessed_subj.loc[i,'edgeRT_log'] = np.log(np.float64(data_subj.loc[i,'edgeRT']))
            preprocessed_subj.loc[i,'edgeRT_log_zscore'] = (np.log(np.float64(data_subj.loc[i,'edgeRT'])) - np.log((data_subj['edgeRT'].astype(float)).mean())) / np.log(data_subj['edgeRT'].astype(float)).std()
            preprocessed_subj.loc[i,'rt'] = data_subj.loc[i,'rt']
            preprocessed_subj.loc[i,'skip'] = data_subj.loc[i,'skip']
            preprocessed_subj.loc[i,'edgeClickLog'] = data_subj.loc[i,'edgeClickLog']
            preprocessed_subj.loc[i,'edgeClickNum'] = len(eval(data_subj.loc[i,'edgeClickLog']))
            preprocessed_subj.at[i,'edge'] = stringtoset(data_subj.loc[i,'edge'])  
    
            if (data_subj.condition[i] == 'normal'):
                preprocessed_subj.at[i,'inference_edge'] = 0
                preprocessed_subj.loc[i,'inference_edgeClickLog'] = 0
                preprocessed_subj.loc[i,'inference_edgeClickNum'] = 0
                preprocessed_subj.loc[i,'inference_edgeRT_zscore'] = 0
                preprocessed_subj.loc[i,'inference_edgeRT_log'] = 0
                preprocessed_subj.loc[i,'inference_edgeRT_log_zscore'] = 0
                preprocessed_subj.loc[i,'inference_edgeRT'] = 0
            else:
                preprocessed_subj.at[i,'inference_edge'] = [stringtoset(i) for i in data_subj.loc[i,'inference_edge'].split(',')]
                preprocessed_subj.loc[i,'inference_edgeClickLog'] = data_subj.loc[i,'inference_edgeClickLog']
                preprocessed_subj.loc[i,'inference_edgeClickNum'] = len(eval(data_subj.loc[i,'inference_edgeClickLog'])) if data_subj.trial_id[i].startswith('test') else "training"  
                preprocessed_subj.loc[i,'inference_edgeRT_zscore'] = (np.float64(data_subj.loc[i, 'inference_edgeRT']) - (data_subj['inference_edgeRT'].astype(float).mean())) / (data_subj['inference_edgeRT'].astype(float).std())
                preprocessed_subj.loc[i,'inference_edgeRT_log'] = np.log(np.float64(data_subj.loc[i,'inference_edgeRT']))
                preprocessed_subj.loc[i,'inference_edgeRT_log_zscore'] = (np.log(np.float64(data_subj.loc[i,'inference_edgeRT'])) - np.log((data_subj['inference_edgeRT'].astype(float)).mean())) / np.log(data_subj['inference_edgeRT'].astype(float)).std()
                preprocessed_subj.loc[i,'inference_edgeRT'] = np.float64(data_subj.loc[i,'inference_edgeRT'])


            if((data_subj.loc[i,'flipped']=='True') | (data_subj.loc[i,'flipped']=='TRUE')):# flip back the flipped trials
                preprocessed_subj.at[i,'edge'] = flipper[stringtoset(data_subj.loc[i,'edge'])]
                if (preprocessed_subj.loc[i,'condition'] != 'normal'):
                    # print(preprocessed_subj.at[i,'inference_edge'] )
                    preprocessed_subj.at[i,'inference_edge'] = [flipper[i] for i in preprocessed_subj.at[i,'inference_edge']]
                    preprocessed_subj.at[i,'inference_edge'] 
            if data_subj.loc[1,'congruency']=='C':
                preprocessed_subj.at[i,'group'] = 'CI'
            else:
                preprocessed_subj.at[i,'group'] = 'II'
            # print(preprocessed_subj.loc[i,'subjID'],preprocessed_subj.condition[i],preprocessed_subj.loc[i,'inference_edge'])
        #       trial type collapses across flipped so we can look at avg across both
        # preprocessed_subj['trialtype'] = [i[:8] if(i[0]=='i') else i[:6] for i in preprocessed_subj.trial_id]
        preprocessed_subj['trialtype'] = preprocessed_subj['trial_id'].str.replace('_flipped', '', regex=False)

        preprocessed_subj.edgeRT = pd.to_numeric(preprocessed_subj.edgeRT)

        # clear out repeated trial values from some exp bug
        preprocessed_subj = preprocessed_subj.drop_duplicates(subset='index').reset_index(drop=True).copy()

        meanRT = np.sum(preprocessed_subj.edgeRT)
        meanRT_log = np.sum(preprocessed_subj.edgeRT_log)

        for i in preprocessed_subj.index:
            # match by trial type
            trial = simdata.loc[simdata.trial_id == preprocessed_subj['trialtype'][i]]
            preprocessed_subj.loc[i,'edgeRT_norm'] = preprocessed_subj.loc[i,'edgeRT']/meanRT
            preprocessed_subj.loc[i,'edgeRT_log_norm'] = preprocessed_subj.loc[i,'edgeRT_log']/meanRT
            
            #get model utility for resp  
            preprocessed_subj.at[i,'U_obm'] = trial['U_obm'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_obm'] =  preprocessed_subj.loc[i,'U_obm'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_obm_max'] = trial['U_obm_max'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_obm_max'] =  preprocessed_subj.loc[i,'U_obm_max'][preprocessed_subj.loc[i,'edge']]
            
            preprocessed_subj.at[i,'U_nbm'] = trial['U_nbm'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_nbm'] =  preprocessed_subj.loc[i,'U_nbm'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_fr'] = trial['U_fr'].tolist()[0]
            preprocessed_subj.loc[i,'edge_resp_U_fr'] =  preprocessed_subj.loc[i,'U_fr'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_rand'] = trial['U_rand'].tolist()[0]
            preprocessed_subj.loc[i,'edge_utility_U_rand'] =  preprocessed_subj.loc[i,'U_rand'][preprocessed_subj.loc[i,'edge']]

            preprocessed_subj.at[i,'U_obm_max_edge'] = max_value_keys(preprocessed_subj.loc[i,'U_obm'])[0]
            
            #get max model ulitliy
            preprocessed_subj.loc[i,'edge_max_U_obm'] =  preprocessed_subj.loc[i,'U_obm'][max_value_keys(preprocessed_subj.loc[i,'U_obm'])[0]]
            preprocessed_subj.loc[i,'edge_max_U_nbm'] =  preprocessed_subj.loc[i,'U_nbm'][max_value_keys(preprocessed_subj.loc[i,'U_nbm'])[0]]
            preprocessed_subj.loc[i,'edge_max_U_fr'] =  preprocessed_subj.loc[i,'U_fr'][max_value_keys(preprocessed_subj.loc[i,'U_fr'])[0]]
            preprocessed_subj.loc[i,'edge_max_U_rand'] =  preprocessed_subj.loc[i,'U_rand'][max_value_keys(preprocessed_subj.loc[i,'U_rand'])[0]]
            
            preprocessed_subj.loc[i,'edge_normresp_U_obm'] =  np.divide(preprocessed_subj.loc[i,'edge_resp_U_obm'],preprocessed_subj.loc[i,'edge_max_U_obm'])
            preprocessed_subj.loc[i,'edge_normresp_U_fr'] =  np.divide(preprocessed_subj.loc[i,'edge_resp_U_fr'],preprocessed_subj.loc[i,'edge_max_U_fr'])
            
            #inf prob
            preprocessed_subj.at[i,'inf_prob'] = trial['inf_prob'].tolist()[0]
            if (preprocessed_subj.condition[i] == 'normal'):
                preprocessed_subj.at[i,'inf_prob_max'] = 0
                preprocessed_subj.at[i,'inf_prob_resp'] = 0
                preprocessed_subj.at[i,'inf_prob_norm'] = 0
                
            else: 
                preprocessed_subj.at[i,'inf_prob_max'] = sum(sorted(preprocessed_subj.at[i,'inf_prob'].values(), reverse=True)[:3])
                preprocessed_subj.at[i,'inf_prob_resp'] = sum([preprocessed_subj.at[i,'inf_prob'][k] for k in preprocessed_subj.at[i,'inference_edge']])
                preprocessed_subj.at[i,'inf_prob_norm'] =  preprocessed_subj.at[i,'inf_prob_resp']/preprocessed_subj.at[i,'inf_prob_max']
                

        #     #quick dirty way to find max bonus pay
            preprocessed_subj.loc[i,'maxbonuscalc'] =  preprocessed_subj.loc[i,'U_obm'][max_value_keys(preprocessed_subj.loc[i,'U_obm'])[0]]

            
            rewards = intTocoord(trial.goal_values.tolist()[0])

            traj = get_traj_from_connections(set(intTocoord(trial.connections.tolist()[0])))

            preprocessed_subj.at[i,'traj'] = traj
            # get features
            preprocessed_subj.at[i,'feature_levels'] = coordToint(get_feature_levels(true_graph))
            preprocessed_subj.at[i,'feature_reward_sum'] = coordToint(get_feature_reward_sum_rmTraj(rewards,true_graph,traj))

            
        #compute bonus pay
        bonus[preprocessed_subj['subjID'].unique()[0]]= sum(preprocessed_subj['edge_resp_U_obm'])


        #total time
        task_dur[fulldata_subj['subjID'].unique()[0]] = fulldata_subj.iloc[-1]['time_elapsed']/60000

        # append to one big df
        fulldata = pd.concat([fulldata,fulldata_subj])
        data = pd.concat([fulldata,fulldata_subj])
        preprocessed = pd.concat([preprocessed,preprocessed_subj])
        preprocessed = preprocessed.reset_index(drop=True)


    return preprocessed,bonus,task_dur, debrief_q, debrief

def filter_skipped_trials(data):
    n = 5
    testdata= data[-5:]
    testdata_noskip = testdata[testdata.skip == 'false'].reset_index(drop=True)
    n_noskip = len(testdata_noskip)
    
    # n = len(data)
    # data_noskip = data[data.skip == 'false'].reset_index(drop=True)
    # n_noskip = len(data_noskip)
    
    data_noskip = data[data.skip == 'false'].reset_index(drop=True)
    return([n_noskip/n,data_noskip])

def get_traj_from_connections(edges):
    foo = set()
    {foo.update(edge) for edge in edges}
    return sorted(tuple(foo), key=lambda x: x[1],reverse=True)