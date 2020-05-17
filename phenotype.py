#!/usr/bin/env python
# coding: utf-8
#default import and definition

df = pd.read_csv('./train_set_phenotype.csv')
test_df = pd.read_csv('./aim3data_test_set_phenotype.csv')

a = test_df.copy()
num = np.size(a,0)
b = df.copy()
num_simple = np.size(b,0)
    
current_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
current_model.load_state_dict(torch.load('./model_params_noise.pkl',map_location='cpu'))
current_model.eval()

actions_test_opt=[]
for i in test_df.index:
    cur_state = states_test[i]
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    mod_q(q_values_test)
    action_opt = torch.max(q_values_test,0)[1].item()
    actions_test_opt.append(action_opt)
actions_test_opt=np.array(actions_test_opt)

tracjectory=pd.DataFrame()
tracjectory['EMPI']=test_df['EMPI']
#tracjectory['EMPI']=df['EMPI']
tracjectory['PDE5I']=0
tracjectory['PDE5I_actual']=0
tracjectory['Nitrate_etc']=0
tracjectory['Nitrate_etc_actual']=0
tracjectory['Statins']=0
tracjectory['Statins_actual']=0
tracjectory['ACEI/ARB/Beta_blocker']=0
tracjectory['ACEI/ARB/Beta_blocker_actual']=0
for i in range(num):
    cur_state = states_test[i]
    action = actions_test[i]
    #cur_state = states[i]
    #action = actions[i]
    #action = actions_test[j+sq_id].item()

    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    mod_q(q_values_test)
    action_opt = torch.max(q_values_test,0)[1].item()
    actions_test_opt[i]=action_opt
        
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
    elif actions_test[i]==7 or actions_test[i]==8:
        tracjectory.loc[i,'PDE5I_actual']=1
    elif actions_test[i]==9:
        tracjectory.loc[i,'Statins_actual']=1
        tracjectory.loc[i,'Nitrate_etc_actual']=1


patient_num = np.size(pd.unique(a['EMPI'])) 
sq_id = 0
total_change=0

mean_SBP=4.8169
std_SBP=0.1583
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
        
        if action==action_opt:#actions_test[j+sq_id]==actions_test_opt[j+sq_id]:
            continue
        change=change+1
        total_change=total_change+1
        
    change_rate=change/maxbloc
    for j in range(maxbloc):
        tracjectory.loc[j+sq_id,'change_rate']= change_rate
        tracjectory.loc[j+sq_id,'SBP']= math.exp(states_test[j+sq_id][6]*std_SBP+mean_SBP)-0.1
        tracjectory.loc[j+sq_id,'EGFR']= math.exp(states_test[j+sq_id][27]*std_EGFR+mean_EGFR)-0.1
    
    sq_id = sq_id + maxbloc
  
states_test1=states_test[:5753] #num1 = 5753
actions_test1=actions_test[:5753]
actions_test1_opt=actions_test_opt[:5753]

max_bloc=53
drug_type1=np.zeros([4,max_bloc])
drug_type1_actual=np.zeros([4,max_bloc])
for i in range(num1):
    bloc=int(round(states_test1[i][-1]*53))-1
    if actions_test1_opt[i]==1:
        drug_type1[3,bloc]=drug_type1[3,bloc]+1
    elif actions_test1_opt[i]==2:
        drug_type1[3,bloc]=drug_type1[3,bloc]+1
        drug_type1[2,bloc]=drug_type1[2,bloc]+1
    elif actions_test1_opt[i]==3:
        drug_type1[2,bloc]=drug_type1[2,bloc]+1
    elif actions_test1_opt[i]==4:
        drug_type1[3,bloc]=drug_type1[3,bloc]+1
        drug_type1[2,bloc]=drug_type1[2,bloc]+1
        drug_type1[1,bloc]=drug_type1[1,bloc]+1
    elif actions_test1_opt[i]==5:
        drug_type1[1,bloc]=drug_type1[1,bloc]+1
    elif actions_test1_opt[i]==6:
        drug_type1[1,bloc]=drug_type1[1,bloc]+1
        drug_type1[3,bloc]=drug_type1[3,bloc]+1
    elif actions_test1_opt[i]==7 or actions_test1_opt[i]==8:
        drug_type1[0,bloc]=drug_type1[0,bloc]+1
    elif actions_test1_opt[i]==9:
        drug_type1[1,bloc]=drug_type1[1,bloc]+1
        drug_type1[2,bloc]=drug_type1[2,bloc]+1
    
    if actions_test1[i]==1:
        drug_type1_actual[3,bloc]=drug_type1_actual[3,bloc]+1
    elif actions_test1[i]==2:
        drug_type1_actual[3,bloc]=drug_type1_actual[3,bloc]+1
        drug_type1_actual[2,bloc]=drug_type1_actual[2,bloc]+1
    elif actions_test1[i]==3:
        drug_type1_actual[2,bloc]=drug_type1_actual[2,bloc]+1
    elif actions_test1[i]==4:
        drug_type1_actual[3,bloc]=drug_type1_actual[3,bloc]+1
        drug_type1_actual[2,bloc]=drug_type1_actual[2,bloc]+1
        drug_type1_actual[1,bloc]=drug_type1_actual[1,bloc]+1
    elif actions_test1[i]==5:
        drug_type1_actual[1,bloc]=drug_type1_actual[1,bloc]+1
    elif actions_test1[i]==6:
        drug_type1_actual[1,bloc]=drug_type1_actual[1,bloc]+1
        drug_type1_actual[3,bloc]=drug_type1_actual[3,bloc]+1
    elif actions_test1[i]==7 or actions_test1[i]==8:
        drug_type1_actual[0,bloc]=drug_type1_actual[0,bloc]+1
    elif actions_test1[i]==9:
        drug_type1_actual[1,bloc]=drug_type1_actual[1,bloc]+1
        drug_type1_actual[2,bloc]=drug_type1_actual[2,bloc]+1
    

        drug_type1_actual_whole[1,bloc]=drug_type1_actual_whole[1,bloc]+1
    elif actions1_whole[i]==6:
        drug_type1_actual_whole[1,bloc]=drug_type1_actual_whole[1,bloc]+1
        drug_type1_actual_whole[3,bloc]=drug_type1_actual_whole[3,bloc]+1
    elif actions1_whole[i]==7 or actions1_whole[i]==8:
        drug_type1_actual_whole[0,bloc]=drug_type1_actual_whole[0,bloc]+1
    elif actions1_whole[i]==9:
        drug_type1_actual_whole[1,bloc]=drug_type1_actual_whole[1,bloc]+1
        drug_type1_actual_whole[2,bloc]=drug_type1_actual_whole[2,bloc]+1
