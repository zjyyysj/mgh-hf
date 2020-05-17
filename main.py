from pyflann import FLANN
import torch 
    
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from itertools import count
import copy
import numpy as np
import math
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame

from nec_agent import NECAgent
#from utils.atari_wrapper import make_atari, wrap_deepmind

%load_ext autoreload
%autoreload 2

class Embed(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super(Embed, self).__init__()
        self.fc1 = nn.Linear(num_inputs,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,num_outputs)
        

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
df = pd.read_csv('HFpEF data/aim3data_train_set.csv')
test_df = pd.read_csv('HFpEF data/aim3data_test_set.csv')

feature_fields=['Gender', 'Age', 'Height', 'BW', 'BMI', 'BSA', 'SBP', 'DBP',
       'Pulse', 'O2_saturation', 'AMI', 'CVD', 'Cardiomyopathy', 'DM', 'HT',
       'Obesity', 'Chronic_Kidney_Disease', 'Arrhythmia (broad Rapid)',
       'Arrhythmia (broad Slow)', 'Atrial_Fibrillation',
       'Pulmonary_Hypertension', 'Anemia', 'Metabolic syndrome', 'Neoplasm',
       'Chemotherapy', 'NT-proBN', 'BNP', 'EGFR', 'CRP_HS', 'ESR', 'TROPONI_I',
       'TROPONIN_T', 'CK_MB', 'ALT', 'AST', 'BUN', 'PROTEIN', 'HDL', 'LDL',
       'WBC', 'HGB', 'PLT', 'RDW', 'pH', 'PO2', 'PCO2', 'FIO2', 'LVEDV',
       'LVEDVI', 'LVESV', 'LVESVI', 'LVEDD', 'LVESD', 'AS', 'PL', 'LVEF',
       'LVSV', 'CO', 'LVMASS', 'LVMASSI', 'RVEDV', 'RVEDVI', 'RVESV', 'RVESVI',
       'RVSV', 'RVEF', 'LA_Anteroposterior_Dimension',
       'RestRegionalWallMotion', 'RestPerfusion', 'LVLateEnhancemen',
       'Diuretics', 'MRA', 'PDE5I', 'Nitrate_etc', 'PKG-Stimulating_drugs',
       'Inotropics', 'Statins', 'Beta_blocker', 'ARB', 'ACEI', 'CCB',
       'Radiofrequency_catheter_ablation_for_atrial_fibrillation',
       'Cardioversion', 'PCI', 'CABG', 'HFpEF', 'bloc']

def outcome_reward(day, base=15, c0=0.04, c1=0.001):
    reward=0
    if day>0:
        if day>370:
            reward=-base+c0*370+c1*(day-370)
        else:
            reward=-base+c0*day
    else:
        reward=base-c1*day
    
    return reward

def add_reward(inp_data, base=15, c0=0.04, c1=0.001):
    inp_data['reward'] = 0
    for i in inp_data.index:
        if i == 0:
            continue
        else:
            if inp_data.loc[i, 'EMPI'] != inp_data.loc[i-1, 'EMPI']:
                day=inp_data.loc[i-1, 'Readmission']
                inp_data.loc[i-1,'reward'] = outcome_reward(day)
                
    day=inp_data.loc[len(inp_data)-1, 'Readmission']
    inp_data.loc[len(inp_data)-1, 'reward'] = outcome_reward(day)

def intermediate_reward(state, next_state, c0=1, c1=0.8, c2=0.6, c3=0.4, c4=0.2):
    reward = 0
    if state[27]>=-1.8805 and next_state[27]>=-1.8805:
        reward = reward - c0*(next_state[27]-state[27]) 
    
    if state[26]>=-1.5194 and next_state[26]>=-1.5194:
        reward = reward - c0*(next_state[26]-state[26])
    
    reward=reward-c1*(next_state[4]-state[4])
    
    if state[10]<=-0.5289 and next_state[10]<=-0.5289:
        reward = reward + c2*(next_state[10]-state[10])
        
    if state[9]<=-1.1153 and next_state[9]<=-1.1153:
        reward = reward + c3*(next_state[9]-state[9])
    if state[9]>=-1.3881 and next_state[9]>=-1.3881:
        reward = reward - c3*(next_state[9]-state[9])
    
    if state[7]<=-1.9965 and next_state[7]<=-1.9965:
        reward = reward + c4*(next_state[7]-state[7])
    if state[7]>=-0.3250 and next_state[7]>=-0.3250:
        reward = reward - c4*(next_state[7]-state[7])
        
    if state[8]<=-3.0198 and next_state[8]<=-3.0198:
        reward = reward + c4*(next_state[8]-state[8])
    if state[8]>=-1.7302 and next_state[8]>=-1.7302:
        reward = reward - c4*(next_state[8]-state[8])
        
    return reward

add_reward(df)

class Pseudo_env(object):
    def __init__(self,df,mode="random"):
        self.df=df.copy()
        self.cur=0
        self.mode="random"
        self.cur_state=None
        self.action_space_n=10
    def reset(self):
        if self.mode=="random":
            self.cur=np.random.randint(df.shape[0])
        self.cur_state=self.df.loc[self.cur,feature_fields].values
        return self.cur_state
    def step(self):
        
        action =int(self.df.loc[self.cur, 'Action'])
        reward = self.df.loc[self.cur,'reward']

        done=1
        if self.cur != df.index[-1]:
            # if not terminal step in trajectory 
            #if df.loc[i+1,'bloc'].item() - df.loc[i,'bloc'].item() > 1:
                #return process_sample(add_reward=True, train=True, eval_type = None)
            
            if df.loc[self.cur, 'EMPI'] == df.loc[self.cur+1, 'EMPI']:
                next_state = self.df.loc[self.cur + 1, feature_fields].values
                reward=reward+intermediate_reward(self.cur_state, next_state)
                done= 0
                self.cur+=1
            else:
                # trajectory is finished
                next_state = np.zeros(len(self.cur_state))
                done = 1
        self.cur_state=next_state
        
        return next_state,reward,done,action

env=Pseudo_env(df)
embedding_model = Embed(len(feature_fields),32)
agent = NECAgent(env, embedding_model,batch_size=32,sgd_lr=1e-5)

for t in count():
    if t < 100:
        reward = agent.warmup()
    else:
        reward = agent.episode()
    print("Episode {}\nTotal Reward: {}".format(t, reward))


test_df = pd.read_csv('HFpEF data/aim3data_test_set.csv')
a = test_df.copy()
num = np.size(a,0)
patient_num = np.size(pd.unique(a['EMPI']))

from torch import Tensor
from torch.autograd import Variable

embedding_model.eval()

import pickle
filename = 'evaluate data/aim3state_test.data'
f1 = open(filename,'rb')
states_test = pickle.load(f1)
filename = 'evaluate data/aim3action_test.data'
f2 = open(filename,'rb')
actions_test = pickle.load(f2)

num_action_same_top5 = 0
num_action_same_top3 = 0
num_action_same = 0
idx_action_same = np.zeros(num)

for i in range(num):
    state=states_test[i]
    action=actions_test[i]
    
    state_embedding = embedding_model(Variable(Tensor(state)).unsqueeze(0))
    q_estimates = [dnd.lookup(state_embedding) for dnd in agent.dnd_list]
    
    if action==torch.cat(q_estimates).sort(descending=True)[1][0].item() or action==torch.cat(q_estimates).sort(descending=True)[1][1].item() or action==torch.cat(q_estimates).sort(descending=True)[1][2].item():
        num_action_same_top3 = num_action_same_top3 + 1
        idx_action_same[i]=3
        if action==torch.cat(q_estimates).max(0)[1].item():
            num_action_same = num_action_same + 1
            idx_action_same[i]=1

ratio = [num_action_same/num, num_action_same_top3/num, num_action_same_top5/num]
print(ratio)
