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
#device=torch.device("cpu")
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

feature_fields=['Gender', 'Age', 'Height', 'BW', 'BMI', 'BSA', 'SBP/DBP', 'DBP',
       'Pulse', 'O2_saturation', 'AMI', 'CVD', 'Cardiomyopathy', 'DM', 'HT',
       'Obesity', 'Chronic_Kidney_Disease', 'Arrhythmia (broad Rapid)',
       'Arrhythmia (broad Slow)', 'Atrial_Fibrillation',
       'Pulmonary_Hypertension', 'Anemia', 'Metabolic syndrome', 'Neoplasm',
       'Chemotherapy', 'NT-proBNP', 'BNP/NT-proBNP', 'EGFR', 'CRP_HS', 'ESR', 'TROPONI_I',
       'TROPONIN_T', 'CK_MB', 'ALT', 'AST', 'BUN', 'PROTEIN', 'HDL', 'LDL',
       'WBC', 'HGB', 'PLT', 'CRP_HS', 'pH', 'PO2/FIO2', 'PCO2', 'FIO2', 'LVEDV',
       'LVEDVI', 'LVESV/LVESD', 'LVESVI', 'LVEDD', 'LVESD', 'AS', 'PL', 'LVEF',
       'LVSV', 'RVEDVI', 'LVMASS', 'LVMASSI', 'RVEDV', 'CO', 'RVESV', 'RVESVI',
       'RVSV', 'RVEF', 'LA_Anteroposterior_Dimension',
       'RestRegionalWallMotion', 'RestPerfusion', 'LVLateEnhancemen',
       'Diuretics', 'MRA', 'PDE5I', 'Nitrate_etc', 'PKG-Stimulating_drugs',
       'Inotropics', 'Statins', 'Beta_blocker', 'ARB', 'ACEI', 'CCB',
       'Radiofrequency_catheter_ablation_for_atrial_fibrillation',
       'Cardioversion', 'PCI', 'CABG', 'HFpEF', 'bloc']

#df = pd.read_csv('HFpEF data/aim3data_simple_v1.csv')
test_df = pd.read_csv('./test_set.csv')

REWARD_THRESHOLD = 20

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, use_cuda, std_init=0.4):
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

from time import time
import sklearn
from matplotlib.cm import rainbow
#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

import pickle
filename = './stest.data'
f1 = open(filename,'rb')
states = pickle.load(f1)
filename = './atest.data'
f2 = open(filename,'rb')
actions = pickle.load(f2)

X=states
y = actions

X0 = X
X0=np.delete(X0, 86, axis=1)
X0=np.delete(X0, 80, axis=1)
X0=np.delete(X0, 79, axis=1)
X0=np.delete(X0, 78, axis=1)
X0=np.delete(X0, 77, axis=1)
X0=np.delete(X0, 76, axis=1)
X0=np.delete(X0, 75, axis=1)
X0=np.delete(X0, 74, axis=1)
X0=np.delete(X0, 73, axis=1)
X0=np.delete(X0, 72, axis=1)
X0=np.delete(X0, 71, axis=1)
X0=np.delete(X0, 70, axis=1)
np.size(X0,1)

feature_fields_cut = np.array(feature_fields)
feature_fields_cut = np.delete(feature_fields_cut, 86)
feature_fields_cut = np.delete(feature_fields_cut, 80)
feature_fields_cut = np.delete(feature_fields_cut, 79)
feature_fields_cut = np.delete(feature_fields_cut, 78)
feature_fields_cut = np.delete(feature_fields_cut, 77)
feature_fields_cut = np.delete(feature_fields_cut, 76)
feature_fields_cut = np.delete(feature_fields_cut, 75)
feature_fields_cut = np.delete(feature_fields_cut, 74)
feature_fields_cut = np.delete(feature_fields_cut, 73)
feature_fields_cut = np.delete(feature_fields_cut, 72)
feature_fields_cut = np.delete(feature_fields_cut, 71)
feature_fields_cut = np.delete(feature_fields_cut, 70)
feature_fields_cut

random_forest_classifier = RandomForestClassifier()
param_grid = {
    'n_estimators': [100,200],
    'max_depth':[10,50],
    'min_samples_split':[2,4],
    'max_features': ['sqrt', 'log2']
    
}

grid = GridSearchCV(random_forest_classifier, param_grid = param_grid, cv = 5, verbose = 5, n_jobs = -1)
grid.fit(X0, y)

best_estimator = grid.best_estimator_

n_jobs = -1
# Build a forest and compute the pixel importances
print("Fitting RandomForestClassifier on faces data with %d cores..." % n_jobs)
t0 = time()
forest_cli = RandomForestClassifier(max_depth=10, max_features='log2', min_samples_split=4, n_estimators=500, n_jobs = -1)

forest_cli.fit(X0, y)
print("done in %0.3fs" % (time() - t0))

importances_cli = forest_cli.feature_importances_
indices_cli = np.argsort(importances_cli)[::-1]
