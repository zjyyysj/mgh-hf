#!/usr/bin/env python
# coding: utf-8

# default import and definition

test_df = pd.read_csv('./test_set.csv')

a = test_df.copy()
num = np.size(a,0)

import pickle
filename = './state_test.data'
f1 = open(filename,'rb')
states_test = pickle.load(f1)
filename = './optaction_test.data'
f1 = open(filename,'rb')
actions_test_opt = pickle.load(f1)


current_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
current_model.load_state_dict(torch.load('./model_params_noise.pkl',map_location='cpu'))
current_model.eval()

tracjectory=pd.DataFrame()
tracjectory['EMPI']=test_df['EMPI']
tracjectory['PDE5I']=0
tracjectory['PDE5I_actual']=0
tracjectory['Nitrate_etc']=0
tracjectory['Nitrate_etc_actual']=0
tracjectory['Statins']=0
tracjectory['Statins_actual']=0
tracjectory['ACEI/ARB/Beta_blocker']=0
tracjectory['ACEI/ARB/Beta_blocker_actual']=0
tracjectory['bloc']=0
actions_test_opt=np.zeros(num)
for i in range(num):
    cur_state = states_test[i]
    action = actions_test[i]
    #action = actions_test[j+sq_id].item()

    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    mod_q(q_values_test)
    action_opt = torch.max(q_values_test,0)[1].item()
    actions_test_opt[i]=action_opt
    tracjectory.loc[i,'bloc']=int(round(cur_state[-1]*53))
    
    if actions_test_opt[i]==1:
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker']=1
    elif actions_test_opt[i]==2:
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker']=1
        tracjectory.loc[i,'Statins']=1
    elif actions_test_opt[i]==3:
        tracjectory.loc[i,'Statins']=1
    elif actions_test_opt[i]==4:
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker']=1
        tracjectory.loc[i,'Statins']=1
        tracjectory.loc[i,'Nitrate_etc']=1
    elif actions_test_opt[i]==5:
        tracjectory.loc[i,'Nitrate_etc']=1
    elif actions_test_opt[i]==6:
        tracjectory.loc[i,'Nitrate_etc']=1
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker']=1
    elif actions_test_opt[i]==7 or actions_test_opt[i]==8:
        tracjectory.loc[i,'PDE5I']=1
    elif actions_test_opt[i]==9:
        tracjectory.loc[i,'Statins']=1
        tracjectory.loc[i,'Nitrate_etc']=1
        
    if actions_test[i]==1:
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker_actual']=1
    elif actions_test[i]==2:
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker_actual']=1
        tracjectory.loc[i,'Statins_actual']=1
    elif actions_test[i]==3:
        tracjectory.loc[i,'Statins_actual']=1
    elif actions_test[i]==4:
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker_actual']=1
        tracjectory.loc[i,'Statins_actual']=1
        tracjectory.loc[i,'Nitrate_etc_actual']=1
    elif actions_test[i]==5:
        tracjectory.loc[i,'Nitrate_etc_actual']=1
    elif actions_test[i]==6:
        tracjectory.loc[i,'Nitrate_etc_actual']=1
        tracjectory.loc[i,'ACEI/ARB/Beta_blocker_actual']=1
    elif actions_test[i]==7 or actions_test_opt[i]==8:
        tracjectory.loc[i,'PDE5I_actual']=1
    elif actions_test[i]==9:
        tracjectory.loc[i,'Statins_actual']=1
        tracjectory.loc[i,'Nitrate_etc_actual']=1


patient_num = np.size(pd.unique(a['EMPI'])) #280 distinct patients in test set
sq_id = 0
total_change=0

mean_SBP=4.8169
std_SBP=0.1583
mean_DBP=4.768677
std_DBP=0.222751

std_Pulse=0.203787
mean_Pulse=4.323302

std_LDL=0.4424
mean_LDL=4.2945

#mean_pH=7.4110
#std_pH=0.0795
mean_Weight=84.23316
std_Weight=23.3062

mean_TROPONI_I=-1.0722
std_TROPONI_I=1.0721
mean_CK_MB=1.3571
std_CK_MB=0.9868
mean_EGFR=3.7210
std_EGFR=0.5570
for i in range(patient_num):
    
    empi = pd.unique(a['EMPI'])[i]
    maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
    patient = a[a['EMPI'] == empi]

    change=0
    for j in range(maxbloc):
        cur_state = states_test[j+sq_id]
        action = actions_test[j+sq_id]
        #action = actions_test[j+sq_id].item()

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        mod_q(q_values_test)
        action_opt = torch.max(q_values_test,0)[1].item()
        #action_opt=actions_test_opt[j+sq_id]
        if action==action_opt:#actions_test[j+sq_id]==actions_test_opt[j+sq_id]:
            continue
        change=change+1
        total_change=total_change+1
        
    change_rate=change/maxbloc
    for j in range(maxbloc):
        tracjectory.loc[j+sq_id,'change_rate']= change_rate
        tracjectory.loc[j+sq_id,'SBP']= math.exp(states_test[j+sq_id][6]*std_SBP+mean_SBP)-0.1
        tracjectory.loc[j+sq_id,'DBP']= math.exp(states_test[j+sq_id][7]*std_DBP+mean_DBP)-0.1
        tracjectory.loc[j+sq_id,'Pulse']= math.exp(states_test[j+sq_id][8]*std_Pulse+mean_Pulse)-0.1
        tracjectory.loc[j+sq_id,'AMI']= states_test[j+sq_id][10]+0.5
        tracjectory.loc[j+sq_id,'CVD']= states_test[j+sq_id][11]+0.5
        tracjectory.loc[j+sq_id,'DM']= states_test[j+sq_id][13]+0.5
        tracjectory.loc[j+sq_id,'LDL']= math.exp(states_test[j+sq_id][38]*std_LDL+mean_LDL)-0.1
        tracjectory.loc[j+sq_id,'Pulmonary_Hypertension']= states_test[j+sq_id][20]+0.5 
        tracjectory.loc[j+sq_id,'Weight']= states_test[j+sq_id][4]*std_Weight+mean_Weight #no log
        tracjectory.loc[j+sq_id,'TROPONI_I']= math.exp(states_test[j+sq_id][30]*std_CK_MB+mean_CK_MB)-0.1
        tracjectory.loc[j+sq_id,'CK_MB']= math.exp(states_test[j+sq_id][32]*std_CK_MB+mean_CK_MB)-0.1
        tracjectory.loc[j+sq_id,'EGFR']= math.exp(states_test[j+sq_id][27]*std_EGFR+mean_EGFR)-0.1
        tracjectory.loc[j+sq_id,'Diuretics']= states_test[j+sq_id][70]+0.5
    
    sq_id = sq_id + maxbloc


