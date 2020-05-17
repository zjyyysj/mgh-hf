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

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

import replay_buffer
from replay_buffer import PrioritizedReplayBuffer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_dir="./tmp"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device=torch.device("cpu")
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

df = pd.read_csv('HFpEF data/aim3data_train_set.csv')
test_df = pd.read_csv('HFpEF data/aim3data_test_set.csv')

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
    
reward_Add(df)

def process_sample(sample_size=1, add_reward=True, train=True, eval_type = None):
    if not train:
        if eval_type is None:
            raise Exception('Provide eval_type to process_batch')
        elif eval_type == 'train':
            a = df.copy()
        elif eval_type == 'val':
            a = val_df.copy()
        elif eval_type == 'test':
            a = test_df.copy()
        else:
            raise Exception('Unknown eval_type')
    else:
        a = df.sample(n=sample_size)
        
    states = None
    actions = None
    rewards = None
    next_states = None
    done_flags = None
    for i in a.index:
        cur_state = a.loc[i,feature_fields]
        action = a.loc[i,'Action']
        reward = a.loc[i,'reward']

        if i != df.index[-1]:
            # if not terminal step in trajectory 
            if df.loc[i+1,'bloc'].item() - df.loc[i,'bloc'].item() > 1:
                return process_sample(add_reward=True, train=True, eval_type = None)
            
            if df.loc[i, 'EMPI'] == df.loc[i+1, 'EMPI']:
                next_state = df.loc[i + 1, feature_fields]
                done_flag = 0
            else:
                # trajectory is finished
                next_state = np.zeros(len(cur_state))
                done_flag = 1
        else:
            # last entry in df is the final state of that trajectory
            next_state = np.zeros(len(cur_state))
            done_flag = 1
                
        if states is None:
            states = copy.deepcopy(cur_state)
        else:
            states = np.vstack((states,cur_state))

        if actions is None:
            actions = [action]
        else:
            actions = np.vstack((actions,action))
        
        if add_reward and done_flag == 0:
            reward = reward + intermediate_reward(cur_state, next_state) # add intermediate reward
        if rewards is None:
            rewards = [reward]
        else:
            rewards = np.vstack((rewards,reward))

        if next_states is None:
            next_states = copy.deepcopy(next_state)
        else:
            next_states = np.vstack((next_states,next_state))

        if done_flags is None:
            done_flags = [done_flag]
        else:
            done_flags = np.vstack((done_flags,done_flag))
    
    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

def do_eval(eval_type):
    pass

current_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
target_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
    
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

beta_start = 0.2

replay_buffer = PrioritizedReplayBuffer(50000, alpha=1)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model, target_model)
REWARD_THRESHOLD=20

def compute_td_loss(batch_size, beta):
    per_epsilon = 1e-5
    state, action, reward, next_state, dones, weights, indices = replay_buffer.sample(batch_size, beta)
    
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    dones      = Variable(torch.FloatTensor(np.float32(dones)))
    weights    = Variable(torch.FloatTensor(weights))
    
    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)
    
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value = next_q_state_values.gather(1,torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    next_q_value=next_q_value.clamp( -REWARD_THRESHOLD , REWARD_THRESHOLD )
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    
    loss = (q_value - expected_q_value.detach()).pow(2) *weights
    prios = loss + per_epsilon
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

max_iters = 150000
sample = 2
batch_size = 32
gamma = 0.99

beta_i= lambda i: min(1.0, beta_start + 1 * i * (1.0 - beta_start)/(max_iters * sample)) 
losses = []
all_losses = []

start=timeit.default_timer()
for i in range(max_iters * sample):
    state, action, reward, next_state, done, sampled_df = process_sample(sample_size=1, add_reward=True)
    replay_buffer.push(state, action, reward, next_state, done)
        
    if len(replay_buffer) > batch_size and i % sample == 0:
        beta = beta_i(i)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.data.item())
        all_losses.append(loss.data.item())
    if i%(100 * sample)==0:
        update_target(current_model, target_model)
    if i % (1000 * sample) == 0 and i>0:
        end=timeit.default_timer()
        av_loss = np.array(losses).mean()
        print("iter:",i)
        print((end-start)/(1000 * sample),"s/iter")
        print("Average loss is ", av_loss)
        losses=[]
        start=timeit.default_timer()
    if (i % (5000 * sample)==0) and i > 0:
        do_eval("?")

torch.save(current_model.state_dict(), 'model_save/model605_params_noise_15iter.pkl')
