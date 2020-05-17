#!/usr/bin/env python
# coding: utf-8

# default import and definition

test_df = pd.read_csv('./test_set.csv')

current_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
current_model.load_state_dict(torch.load('./model_params_noise.pkl',map_location='cpu'))
current_model.eval()

a = test_df.copy()
num = np.size(a,0)

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

ratio = [num_action_same/num, num_action_same_top2/num, num_action_same_top3/num, num_action_same_top5/num]
print(ratio)

action_same = np.zeros(10)
action_seq=np.zeros(10)
action_rate=np.zeros(10)

for i in range(num_simple):
    
    cur_state = states[i]
    action = actions[i].item()
    
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    #mod_q(q_values_test)
    
    for j in range(10):
        if action==j:
            action_seq[j]=action_seq[j]+1
            if abs(action - torch.max(q_values_test,0)[1].item()) == 0:
                action_same[j] = action_same[j] + 1 
            break
            
for i in range(10):
    action_rate[i]=action_same[i]/action_seq[i]
    
reward_Add(a)
plus=dayplus_Add(a)
minus=dayminus_Add(a)

patient_num = np.size(pd.unique(a['EMPI']))
readmission_exist = np.size(a.query('reward==-15'),0)
readmission_rate = readmission_exist/patient_num

readmission_days_ave=plus
nonreadmission_days_ave=minus
readmission = [readmission_rate, plus, minus]

i=0
j=0
sq_id=0
empi = pd.unique(a['EMPI'])[i]
maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
patient = a[a['EMPI'] == empi]
pd.unique(a['EMPI'])
label=pd.DataFrame()
label['EMPI']=pd.unique(a['EMPI'])

patient_num = np.size(pd.unique(a['EMPI'])) #280 distinct patients in test set

patient_close_opt5 = np.zeros(patient_num)
patient_close_opt3 = np.zeros(patient_num)
patient_close_opt2 = np.zeros(patient_num)
patient_close_opt1 = np.zeros(patient_num)

patient_close_zero = np.zeros(patient_num)

patient_read = np.zeros(patient_num)
patient_read_day = np.zeros(patient_num)
patient_nonread_day = np.zeros(patient_num)
#patient_mor90 = np.zeros(patient_num)
patient_gender = np.zeros(patient_num)
patient_age = np.zeros(patient_num)

sq_id = 0
for i in range(patient_num):
    
    empi = pd.unique(a['EMPI'])[i]
    maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
    patient = a[a['EMPI'] == empi]
    
    close_action_sum5 = 0
    close_action_sum3 = 0
    close_action_sum2 = 0
    close_action_sum1 = 0
    
    close_zero_sum = 0
      
    for j in range(maxbloc):
        
        cur_state = states_test[j+sq_id]
        action = actions_test[j+sq_id]
        #action = actions_test[j+sq_id].item()

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        action_opt = torch.max(q_values_test,0)[1].item()
        
        if action in q_values_test.sort(descending=True)[1][0:5]:
            close_action_sum5 = close_action_sum5 + 1
            if action in q_values_test.sort(descending=True)[1][0:3]:
                close_action_sum3 = close_action_sum3 + 1
                if action in q_values_test.sort(descending=True)[1][0:2]:
                    close_action_sum2 = close_action_sum2 + 1
                    if action in q_values_test.sort(descending=True)[1][0:1]:
                        close_action_sum1 = close_action_sum1 + 1
        
        if action == 0:
            close_zero_sum = close_zero_sum + 1
     
    if patient.loc[j+sq_id,'reward']==-15:
        patient_read[i] = 1
        patient_read_day[i] = patient.loc[j+sq_id,'Readmission']
    else:
        patient_nonread_day[i] = -patient.loc[j+sq_id,'Readmission']

    if patient.loc[j+sq_id,'Gender'] > 0:
        patient_gender[i] = 1
    
    patient_age[i] = patient.loc[j+sq_id,'Age']
        
    patient_close_opt5[i] = close_action_sum5/maxbloc
    patient_close_opt3[i] = close_action_sum3/maxbloc
    patient_close_opt2[i] = close_action_sum2/maxbloc
    patient_close_opt1[i] = close_action_sum1/maxbloc
    patient_close_zero[i] = close_zero_sum/maxbloc
    
    sq_id = sq_id + maxbloc

patient_close_zero = np.zeros(patient_num)

patient_read = np.zeros(patient_num)
patient_read_day = np.zeros(patient_num)
patient_nonread_day = np.zeros(patient_num)


sq_id = 0
for i in range(patient_num):
    
    empi = pd.unique(a['EMPI'])[i]
    maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
    patient = a[a['EMPI'] == empi]
    
    
    close_zero_sum = 0
      
    for j in range(maxbloc):
        
        cur_state = states_test[j+sq_id]
        action = actions_test[j+sq_id]
        #action = actions_test[j+sq_id].item()

        if action == 0:
            close_zero_sum = close_zero_sum + 1
     
    if patient.loc[j+sq_id,'reward']==-15:
        patient_read[i] = 1
        patient_read_day[i] = patient.loc[j+sq_id,'Readmission']
    else:
        patient_nonread_day[i] = -patient.loc[j+sq_id,'Readmission']

    patient_close_zero[i] = close_zero_sum/maxbloc
    
    sq_id = sq_id + maxbloc


