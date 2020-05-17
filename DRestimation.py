#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import math
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import pickle
import matplotlib.pyplot as plt
import timeit


import replay_buffer
from replay_buffer import PrioritizedReplayBuffer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_dir="./tmp"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

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

#df = pd.read_csv('./train_set.csv')
test_df = pd.read_csv('./test_set.csv')

REWARD_THRESHOLD = 20
a = test_df.copy()
num = np.size(a,0)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, use_cuda, std_init=0.1):
        super(NoisyLinear, self).__init__()
        
        self.use_cuda     = use_cuda
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.use_cuda:
            weight_epsilon = self.weight_epsilon.cuda()
            bias_epsilon   = self.bias_epsilon.cuda()
        else:
            weight_epsilon = self.weight_epsilon
            bias_epsilon   = self.bias_epsilon
            
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
class DuelingDQNnoise(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQNnoise, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.linear = nn.Linear(num_inputs, 128)
        
        self.noisy_value1 = NoisyLinear(128, 128, use_cuda = USE_CUDA)
        self.noisy_value2 = NoisyLinear(128, 1, use_cuda = USE_CUDA)
        
        self.noisy_advantage1 = NoisyLinear(128, 128, use_cuda = USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(128, self.num_outputs, use_cuda = USE_CUDA)
        
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        x = value + advantage - advantage.mean() 
        return x
    
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
        
    
    def act(self, state):
        with torch.no_grad():
            state   = Variable(torch.FloatTensor(state).unsqueeze(0))
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data.item()
        return action
    
current_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
current_model.load_state_dict(torch.load('./model_params_noise.pkl',map_location='cpu'))
current_model.eval()

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

def reward_Add(inp_data, base=15, c0=0.04, c1=0.001):
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
    
reward_Add(a)
reward_Add(b)

def softmax_prob(q_value, beta = 10):
    t = np.exp( beta * (q_value - np.max(q_value)))
    return t/sum(t)

patient_num = np.size(pd.unique(a['EMPI']))
patient_DR = np.zeros(patient_num)
patient_rhos = []
gamma = 0.99
sq_id = 0

for i in range(patient_num): 
    # extract each patient
    empi = pd.unique(a['EMPI'])[i]
    maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
    patient = a[a['EMPI'] == empi] # extract the trajectory of this patient
    
    # define the necessary elements 
    patient_rho = [1]
    patient_reward = [0]
    #patient_reward_new = [0] 
    patient_gamma = [0]
    patient_q = [0]
    patient_v = []
    
    for j in range(maxbloc):
        
        cur_state = patient.loc[j+sq_id,feature_fields]
        reward = patient.loc[j+sq_id,'reward']
        #reward_new = patient.loc[j+sq_id,'reward_new']
        
        if j < (maxbloc-1):
            if patient.loc[j+sq_id+1,'bloc'].item() - patient.loc[j+sq_id,'bloc'].item() > 1: # ignore the uncontinuous transition
                continue
            else:
                next_state = patient.loc[j+sq_id+1, feature_fields]
                reward = reward + intermediate_reward(cur_state, next_state)
                #reward_new = reward
                
        action=int(df.loc[j+sq_id, 'Action']) 

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        #mod_q(q_values_test)
        
        q_values = q_values_test.cpu().detach().numpy()
        clinician_values_test = clinician_model(state)
        clinician_values = clinician_values_test.cpu().detach().numpy()
        eva_prob = softmax_prob(q_values)[action] # evaluation_prob(action(t) | state(t))
        eva_prob = move_prob(eva_prob)
        cli_prob = move_prob(softmax_prob(clinician_values)[action]) # behavior_prob(action(t) | state(t))
        rho = eva_prob / cli_prob
        rho = cli_prob
        #rho = 1
        
        patient_rho.append(rho*patient_rho[-1])
        patient_rhos.append(rho*patient_rho[-1])
        patient_reward.append(reward)
        #patient_reward_new.append(reward_new)
        patient_gamma.append(gamma**j)
        
        q_value = q_values[action]
        q_value = max(-REWARD_THRESHOLD, min(q_value, REWARD_THRESHOLD)) #clamp the extremely large or small
        patient_q.append(q_value)
        
        value = sum(q_values*softmax_prob(q_values)) # use value = Sum_a softmax(q(s,a)) * q(s,a)
        value = max(-REWARD_THRESHOLD, min(value, REWARD_THRESHOLD))  #clamp the extremely large or small
        patient_v.append(value)
    
    patient_v.append(0)
    patient_rho = np.array(patient_rho)
    patient_reward = np.array(patient_reward)
    #patient_reward_new = np.array(patient_reward_new)
    patient_gamma = np.array(patient_gamma)
    patient_q = np.array(patient_q)
    patient_v = np.array(patient_v)
    
    
    patient_DR[i] = sum(patient_gamma * patient_rho * patient_reward) - sum(patient_gamma*(patient_rho*patient_q - patient_rho*patient_v))
    bound = 100
    patient_DR[i] = max(-bound, min(patient_DR[i], bound))
    sq_id = sq_id + maxbloc

print(len(patient_rhos),np.mean(patient_rhos),np.var(patient_rhos))

