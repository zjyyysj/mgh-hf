#!/usr/bin/env python
# coding: utf-8

# default import and definition
test_df = pd.read_csv('./test_set.csv')
current_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
current_model.load_state_dict(torch.load('./model_params_noise.pkl',map_location='cpu'))
current_model.eval()

a = test_df.copy()
num = np.size(a,0)

import pickle
filename = './state_test.data'
f1 = open(filename,'rb')
states_test = pickle.load(f1)
filename = './action_test.data'
f2 = open(filename,'rb')
actions_test = pickle.load(f2)


num_action_same_top5 = 0
num_action_same_top3 = 0
num_action_same_top2 = 0
num_action_same = 0
idx_action_same = np.zeros(num)

for i in range(num):
    
    cur_state = states_test[i]
    action = actions_test[i].item()
    
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    mod_q(q_values_test)
    
    if action in q_values_test.sort(descending=True)[1][0:5]:
        num_action_same_top5 = num_action_same_top5 + 1
        idx_action_same[i]=5
        if action in q_values_test.sort(descending=True)[1][0:3]:
            num_action_same_top3 = num_action_same_top3 + 1
            idx_action_same[i]=3
            if action in q_values_test.sort(descending=True)[1][0:2]:
                num_action_same_top2 = num_action_same_top2 + 1
                idx_action_same[i]=2
                if abs(action - torch.max(q_values_test,0)[1].item()) == 0:
                    num_action_same = num_action_same + 1
                    idx_action_same[i]=1

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

def reward_Add(inp_data):
    inp_data['reward'] = 0
    for i in inp_data.index:
        if i == 0:
            continue
        else:
            if inp_data.loc[i, 'EMPI'] != inp_data.loc[i-1, 'EMPI']:
                day=inp_data.loc[i-1, 'Readmission']
                inp_data.loc[i-1,'reward'] = outcome_reward(day,15,0,0)
                
    day=inp_data.loc[len(inp_data)-1, 'Readmission']
    inp_data.loc[len(inp_data)-1, 'reward'] = outcome_reward(day,15,0,0)

def dayplus_Add(inp_data):
    inp_data['dayplus'] = 0
    dayplus=0
    patient=0
    avedayplus=0
    for i in inp_data.index:
        if i == 0:
            continue
        else:
            if inp_data.loc[i, 'EMPI'] != inp_data.loc[i-1, 'EMPI']:
                day=inp_data.loc[i-1, 'Readmission']
                if day>0:
                    inp_data.loc[i-1,'dayplus'] = day
                    dayplus=dayplus+day
                    patient=patient+1
                
    day=inp_data.loc[len(inp_data)-1, 'Readmission']
    if day>0:
        inp_data.loc[len(inp_data)-1, 'dayplus'] = day
        dayplus=dayplus+day
        patient=patient+1
    
    if patient>0:
        avedayplus=dayplus/patient

    return avedayplus
    
def dayminus_Add(inp_data):
    inp_data['dayminus'] = 0
    dayminus=0
    patient=0
    avedayminus=0
    for i in inp_data.index:
        if i == 0:
            continue
        else:
            if inp_data.loc[i, 'EMPI'] != inp_data.loc[i-1, 'EMPI']:
                day=inp_data.loc[i-1, 'Readmission']
                if day<0:
                    inp_data.loc[i-1,'dayminus'] = -day
                    dayminus=dayminus+day
                    patient=patient+1
                
    day=inp_data.loc[len(inp_data)-1, 'Readmission']
    if day<0:
        inp_data.loc[len(inp_data)-1, 'dayminus'] = -day
        dayminus=dayminus+day
        patient=patient+1
    if patient>0:
        avedayminus=dayminus/patient
    
    return avedayminus

    
reward_Add(a)
plus=dayplus_Add(a)
minus=dayminus_Add(a)
reward_Add(b)
plus_s=dayplus_Add(b)
minus_s=dayminus_Add(b)

patient_num = np.size(pd.unique(a['EMPI']))
readmission_exist = np.size(a.query('reward==-15'),0)
readmission_rate = readmission_exist/patient_num

readmission_days_ave=plus
nonreadmission_days_ave=minus

patient_num_s = np.size(pd.unique(b['EMPI']))
readmission_exist_s = np.size(b.query('reward==-15'),0)
readmission_rate_s = readmission_exist_s/patient_num_s
readmission_days_ave_s=plus_s
nonreadmission_days_ave_s=minus_s
readmission = [readmission_rate, readmission_rate_s, plus, minus, plus_s, minus_s]
readmission

label=pd.DataFrame()
label['EMPI']=pd.unique(a['EMPI'])

label_s=pd.DataFrame()
label_s['EMPI']=pd.unique(b['EMPI'])

patient_num = np.size(pd.unique(a['EMPI'])) #280 distinct patients in test set
label['change_Action']=0
label['change_FirstIndex']=0
label['readmission']=0
label['maxbloc']=0
label['no_medication']=0
label['change_zero']=0
label['Action1']=0
label['Action2']=0
label['Action3']=0
label['Action4']=0
label['Action5']=0
label['Action6']=0
label['Action7']=0
label['Action8']=0
label['Action9']=0

sq_id = 0
change_patient=0

for i in range(patient_num):
    
    empi = pd.unique(a['EMPI'])[i]
    maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
    patient = a[a['EMPI'] == empi]
    change_state=0
    change_actions=np.zeros(10)
    
    for j in range(maxbloc):
        
        cur_state = states_test[j+sq_id]
        action = actions_test[j+sq_id]
        #action = actions_test[j+sq_id].item()

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        mod_q(q_values_test)
        action_opt = torch.max(q_values_test,0)[1].item()
        action_sec=q_values_test.sort(descending=True)[1][1].item()
        
        if action==0 and action_opt>0:
            label.loc[i,'change_Action']=action_opt
            label.loc[i,'change_FirstIndex']=j
            change_patient=change_patient+1
            break

    no_medication=0
    for j in range(maxbloc):
        
        cur_state = states_test[j+sq_id]
        action = actions_test[j+sq_id]
        #action = actions_test[j+sq_id].item()

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        mod_q(q_values_test)
        action_opt = torch.max(q_values_test,0)[1].item()

        if action==0:
            no_medication=no_medication+1
            if action_opt>0:
                change_state=change_state+1
                change_actions[int(action_opt)]=change_actions[int(action_opt)]+1
     
    label.loc[i,'readmission']=patient.loc[j+sq_id,'Readmission']
    label.loc[i,'maxbloc']=maxbloc
    label.loc[i,'no_medication']=no_medication
    if change_state>0:
        label.loc[i,'change_zero']=change_state/no_medication
        label.loc[i,'Action1']=change_actions[1]/change_state
        label.loc[i,'Action2']=change_actions[2]/change_state
        label.loc[i,'Action3']=change_actions[3]/change_state
        label.loc[i,'Action4']=change_actions[4]/change_state
        label.loc[i,'Action5']=change_actions[5]/change_state
        label.loc[i,'Action6']=change_actions[6]/change_state
        label.loc[i,'Action7']=change_actions[7]/change_state
        label.loc[i,'Action8']=change_actions[8]/change_state
        label.loc[i,'Action9']=change_actions[9]/change_state
    
    sq_id = sq_id + maxbloc

change_rate=change_patient/patient_num
print(change_rate)

patient_num = np.size(pd.unique(a['EMPI'])) #280 distinct patients in test set
#label['change_zero_top2']=0
#label['change_to_second']=0
label['Action1_top2']=0
label['Action2_top2']=0
label['Action3_top2']=0
label['Action4_top2']=0
label['Action5_top2']=0
label['Action6_top2']=0
label['Action7_top2']=0
label['Action8_top2']=0
label['Action9_top2']=0

sq_id = 0
change_patient=0

for i in range(patient_num):
    
    empi = pd.unique(a['EMPI'])[i]
    maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
    patient = a[a['EMPI'] == empi]
    change_state=0
    change_state_sec=0
    change_actions=np.zeros(10)

    no_medication=0
    for j in range(maxbloc):
        
        cur_state = states_test[j+sq_id]
        action = actions_test[j+sq_id]
        #action = actions_test[j+sq_id].item()

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        mod_q(q_values_test)
        action_opt = torch.max(q_values_test,0)[1].item()
        action_sec=q_values_test.sort(descending=True)[1][1].item()
        
        if action==0:
            no_medication=no_medication+1
            if action_opt>0:
                change_state=change_state+1
                change_actions[int(action_opt)]=change_actions[int(action_opt)]+1
            else:
                change_actions[int(action_sec)]=change_actions[int(action_sec)]+1

    if no_medication>0:
        label.loc[i,'Action1_top2']=change_actions[1]/no_medication
        label.loc[i,'Action2_top2']=change_actions[2]/no_medication
        label.loc[i,'Action3_top2']=change_actions[3]/no_medication
        label.loc[i,'Action4_top2']=change_actions[4]/no_medication
        label.loc[i,'Action5_top2']=change_actions[5]/no_medication
        label.loc[i,'Action6_top2']=change_actions[6]/no_medication
        label.loc[i,'Action7_top2']=change_actions[7]/no_medication
        label.loc[i,'Action8_top2']=change_actions[8]/no_medication
        label.loc[i,'Action9_top2']=change_actions[9]/no_medication
    sq_id = sq_id + maxbloc

empi=a.loc[i,'EMPI']
item=np.array(label[label['EMPI']==empi])
    
cur_state = states_test[i]
action = actions_test[i]
#action = actions_test[j+sq_id].item()
state = Variable(torch.FloatTensor(np.float32(cur_state)))
q_values_test = current_model(state)
mod_q(q_values_test)
action_opt = torch.max(q_values_test,0)[1].item()
action_sec=q_values_test.sort(descending=True)[1][1].item()
[action_opt,action_sec,action]


# In[38]:


Nomedication_test=test_df.copy()
Nomedication_test['top1_action50']=0
Nomedication_test['top1_action60']=0
Nomedication_test['top1_action70']=0
Nomedication_test['top1_action80']=0
Nomedication_test['top2_action40']=0
Nomedication_test['top2_action50']=0
Nomedication_test['top2_action60']=0
Nomedication_test['top2_action70']=0
for i in range(num):
    
    empi=a.loc[i,'EMPI']
    item=np.array(label[label['EMPI']==empi])
    
    cur_state = states_test[i]
    action = actions_test[i]
    #action = actions_test[j+sq_id].item()

    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    mod_q(q_values_test)
    action_opt = torch.max(q_values_test,0)[1].item()
    action_sec=q_values_test.sort(descending=True)[1][1].item()
    
    if action==0:
        if action_opt>0:
            rate=item[0][int(action_opt)+6]
            if action_opt>=2 and action_opt<=3:
                rate=item[0][8]+item[0][9]
                action_opt=3
            if action_opt>=4 and action_opt<=6:
                rate=item[0][10]+item[0][11]+item[0][12]
                action_opt=5
            if action_opt>=7 and action_opt<=8:
                rate=item[0][13]+item[0][14]
                action_opt=7
            if rate>0.5:
                Nomedication_test.loc[i,'top1_action50']= action_opt
                if rate>=0.6:
                    Nomedication_test.loc[i,'top1_action60']= action_opt
                    if rate>=0.7:
                        Nomedication_test.loc[i,'top1_action70']= action_opt
                        if rate>=0.8:
                            Nomedication_test.loc[i,'top1_action80']= action_opt
        
        if action_sec>0:
            rate_sec=item[0][int(action_sec)+15]
            if action_sec>=2 and action_sec<=3:
                rate=item[0][17]+item[0][18]
                action_sec=3
            if action_sec>=4 and action_sec<=6:
                rate=item[0][19]+item[0][20]+item[0][21]
                action_sec=5
            if action_sec>=7 and action_sec<=8:
                rate=item[0][22]+item[0][23]
                action_sec=7
            if rate_sec>0.4:
                Nomedication_test.loc[i,'top2_action40']= action_sec
                if rate_sec>=0.5:
                    Nomedication_test.loc[i,'top2_action50']= action_sec
                    if rate_sec>=0.6:
                        Nomedication_test.loc[i,'top2_action60']= action_sec
                        if rate_sec>=0.7:
                            Nomedication_test.loc[i,'top2_action70']= action_sec
        
        if action_opt>0:
            rate_opt=item[0][int(action_opt)+15]
            if action_opt>=2 and action_opt<=3:
                rate=item[0][17]+item[0][18]
                action_opt=3
            if action_opt>=4 and action_opt<=6:
                rate=item[0][19]+item[0][20]+item[0][21]
                action_opt=5
            if action_opt>=7 and action_opt<=8:
                rate=item[0][22]+item[0][23]
                action_opt=7
            if rate_opt>0.4:
                Nomedication_test.loc[i,'top2_action40']= action_opt
                if rate_opt>=0.5:
                    Nomedication_test.loc[i,'top2_action50']= action_opt
                    if rate_opt>=0.6:
                        Nomedication_test.loc[i,'top2_action60']= action_opt
                        if rate_opt>=0.7:
                            Nomedication_test.loc[i,'top2_action70']= action_opt
        
