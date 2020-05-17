#!/usr/bin/env python
# coding: utf-8
#default import and definition

test_df = pd.read_csv('./test_set.csv')

current_model = DuelingDQNnoise(len(feature_fields), 10).to(device)
current_model.load_state_dict(torch.load('./model_params_noise.pkl'))
current_model.eval()

a = test_df.copy()
num = np.size(a,0)

pd.unique(a['EMPI'])
patient_num = np.size(pd.unique(a['EMPI']))

import pickle
filename = './state_test.data'
f1 = open(filename,'rb')
states_test = pickle.load(f1)
filename = './action_test.data'
f2 = open(filename,'rb')
actions_test = pickle.load(f2)


zero_action = 0
for i in range(num):
    if actions_test[i]==0:
        zero_action = zero_action + 1

def clamp_return(v, upthreshold=20, downthreshold=20, stand=0): # dispose of some outlier
    v1=v
    v1 = v1 + stand
    if v1> upthreshold:
        v1 = upthreshold
    if v1< -downthreshold:
        v1 = -downthreshold

    
    return v1

num_action_same_top5 = 0
num_action_same_top3 = 0
num_action_same_top2 = 0
num_action_same = 0
idx_action_same = np.zeros(num)

return_cli_opt = []
return_cli_zero = []
return_cli_random = []
return_opt = []
return_cli =[]
return_zero = []
return_random = []

q_random = 0
for i in range(num):
    cur_state = states_test[i]
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    # q_values_test.detach()
    #action = actions_test[i].item()
    action = actions_test[i]
    
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
                
    opt = clamp_return(torch.max(q_values_test,0)[0].item()) 
    cli = clamp_return(q_values_test[action].item())
    zero = clamp_return(q_values_test[0].item())
    for i in range(10):
        q_random+=clamp_return(q_values_test[random.choice(range(25))].item())
    rand = q_random/10
    
    return_opt.append(opt)
    return_cli.append(cli)
    return_zero.append(zero)
    return_random.append(rand)
    
    return_cli_opt.append(opt-cli) #optimal vs clinician
    return_cli_zero.append(cli-zero) # clinician vs zero-drug
    return_cli_random.append(cli - rand)
    q_random = 0


return_opt1 = []
return_opt2 = []
return_opt3 = []
return_zero = []

for i in range(num):
    cur_state = states_test[i]
    state = Variable(torch.FloatTensor(np.float32(cur_state)))
    q_values_test = current_model(state)
    # q_values_test.detach()
    #action = actions_test[i].item()
    #action = actions_test[i]

    return_opt1.append(clamp_return(q_values_test.sort(descending=True)[0][0].item()))
    return_opt2.append(clamp_return(q_values_test.sort(descending=True)[0][1].item()))
    return_opt3.append(clamp_return(q_values_test.sort(descending=True)[0][2].item()))
    return_zero.append(clamp_return(q_values_test[0].item()))

return_opt1=np.array(return_opt1)
return_opt2=np.array(return_opt2)
return_opt3=np.array(return_opt3)
return_zero=np.array(return_zero)

data=[return_opt1,return_opt2,return_opt3,return_zero]
data_diff=[return_opt1-return_opt2, return_opt1-return_opt3, return_opt1-return_zero]

sq_id = 0
for i in range(patient_num):
    
    empi = pd.unique(a['EMPI'])[i]
    maxbloc = pd.value_counts(a['EMPI'])[empi] # total bloc of the i th patient
    patient = a[a['EMPI'] == empi]
    
    opt_path = []
    cli_path = []
    zero_path = []
    rand_path = []
    
    for j in range(maxbloc):
        
        cur_state = patient.loc[j+sq_id,feature_fields]
        action = actions_test[j+sq_id]

        state = Variable(torch.FloatTensor(np.float32(cur_state)))
        q_values_test = current_model(state)
        
        opt_path.append(clamp_return(torch.max(q_values_test,0)[0].item()))
        cli_path.append(clamp_return(q_values_test[action].item()))
        zero_path.append(clamp_return(q_values_test[0].item()))
        q_random = 0
        for i in range(10):
            q_random+=clamp_return(q_values_test[random.choice(range(25))].item())
        rand_path.append(q_random/10)
     
    opt_patient.append(np.mean(opt_path))
    cli_patient.append(np.mean(cli_path))
    zero_patient.append(np.mean(zero_path))
    rand_patient.append(np.mean(rand_path))
    
    sq_id = sq_id + maxbloc
    
x = np.linspace(1,patient_num,patient_num)
opt_patient = np.array(opt_patient)
cli_patient = np.array(cli_patient)
zero_patient = np.array(zero_patient)
rand_patient = np.array(rand_patient)
optcli_patient = opt_patient - cli_patient
clizero_patient = cli_patient - zero_patient
clirand_patient = cli_patient - rand_patient

