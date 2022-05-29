import sys
import os
import numpy as np
import pandas as pd

import training
from src import slant, slant_plus, aslm, sinn, nn, voter, degroot
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Features, Value, ClassLabel
from distutils.util import strtobool

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import configargparse
from scipy import interpolate
import json
from scipy.interpolate import interp1d


p = configargparse.ArgumentParser()
p.add_argument('--method', type=str, default="SINN", 
               help='Options are "Voter", "DeGroot", "AsLM", "SLANT", "SLANT+", "NN", "SINN"')
p.add_argument('--dataset', type=str, default="synthetic_consensus", 
               help='Options are "synthetic_consensus","synthetic_clustering","synthetic_polarization","sample_twitter_Abortion"')
p.add_argument('--save_dir', type=str, default="output/") 
p.add_argument('--batch_size', type=int, default=256, 
               help='Number of epochs to train for.')
p.add_argument('--hidden_features', type=int, default=1,
               help='Number of units in neural network. $L\in\{8,12,16\}$.')
p.add_argument('--num_hidden_layers', type=int, default=7, 
               help='Number of layers in neural network. $L\in\{3,5,7\}$.')
p.add_argument('--alpha', type=float, default=1.0, 
               help='$\\alpha\in\{0.1,1.0,5.0\}$. ')
p.add_argument('--beta', type=float, default=0.1, 
               help='$\\beta\in\{0.1,1.0,5.0\}$. ')
p.add_argument('--num_epochs', type=int, default=1000)
p.add_argument('--lr', type=float, default=0.001, 
               help='learning rate. default=0.001')
p.add_argument('--K', type=int, default=1, 
               help='dimension of latent space $K\in\{1,2,3\}$. ')
p.add_argument('--type_odm', type=str, default="Powerlaw",
               help='Options are "DeGroot", "Powerlaw", "BCM", "FJ"')
p.add_argument('--flag_profile', type=strtobool, default=False)
p.add_argument('--activation_func', type=str, default='tanh',
               help='Options are "sigmoid", "tanh", "relu", "selu", "softplus", "elu"')
opt = p.parse_args()


def prediction2label(x):
    f_x = np.exp(x) / np.sum(np.exp(x), axis=-1)[:,None]
    label = np.argmax(f_x, axis=-1)
    
    return label


def rolling_matrix(x,window_size=21):
    x = x.flatten()
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n-window_size+1, window_size), strides=(stride,stride) ).copy()


class load_data(Dataset):

    def __init__(self, sequence, num_users, initial_u):
        super().__init__()

        uids = sequence[:, 0]
        times = sequence[:, 2]
        opinions = sequence[:, 1]
        
        self.initial_u = np.array(initial_u)


        history = []
        previous = []
        model_out = []
       
        previous = []
        for iu in range(num_users):
            tmpx = sequence[sequence[:,0]==iu,2]
            tmpy = sequence[sequence[:,0]==iu,1]
            if len(tmpx)>0: 
                tmpx = np.append(tmpx, 1.)
                tmpy = np.append(tmpy, tmpy[-1])
            else:
                tmpx = np.array([0,1])
                tmpy = np.array([initial_u[iu],initial_u[iu]])
            tmpf = interp1d(tmpx, tmpy, kind='next', fill_value="extrapolate")
            previous.append( tmpf(times) )
        previous = np.array(previous).T

        history0 = rolling_matrix(sequence[:,0])
        history1 = rolling_matrix(sequence[:,1])
        history2 = rolling_matrix(sequence[:,2])

        dT = np.stack([history0,history1,history2], axis=-1) 
        history = dT[:,:-1,:]
        model_out = dT[:,-1,:]

        self.previous = np.array(previous)
        self.history = history
        self.model_out = model_out

        self.datanum = len(self.model_out)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):

        history = self.history[idx]
        previous = self.previous[idx]
        model_out = self.model_out[idx]

        return {'history': torch.from_numpy(history).float(), 'previous': torch.from_numpy(previous).float(), 
                'initial': torch.from_numpy(self.initial_u).float(), 
                'ui': torch.from_numpy(model_out[:1]).long(), 'ti': torch.from_numpy(model_out[2:]).float()}, \
               {'opinion': torch.from_numpy(model_out[1:2]).float()}




def evaluate(model, sequence, train_sequence, num_users, initial_u, batch_size, nclasses, val_period):

    test_indices = np.where(sequence[:,2]>=val_period)[0]

    opinions = train_sequence[:, 1]
    uids = train_sequence[:, 0]
    initial_u = np.array(initial_u).reshape(1,-1)


    dfs = []
    for ii in test_indices[20:]: 
        current_time = sequence[ii,2]
        tmphistory = train_sequence[train_sequence[:, 2]<current_time,:]
        if len(tmphistory)<50: continue
        
        prev_u = []
        for iu in range(num_users):
            tmpprev_u = train_sequence[(train_sequence[:, 0]==iu)&(train_sequence[:, 2]<current_time),1] 
            if len(tmpprev_u)>0:  
                prev_u.append(tmpprev_u[-1])
            else:
                prev_u.append(0.)
        prev_u = np.array(prev_u) 

        history = tmphistory[np.newaxis,-50:]
        previous = prev_u[np.newaxis]
        model_out = sequence[np.newaxis,ii]

        model_input = {'history': torch.from_numpy(history).float(), 'previous': torch.from_numpy(previous).float(), 
                       'initial': torch.from_numpy(initial_u).float(), #'pi': torch.from_numpy(pi).float(),
                       'ui': torch.from_numpy(model_out[:,:1]).long(), 'ti': torch.from_numpy(model_out[:,2:]).float()}
        model_output = model(model_input)
        test_pred = model_output['opinion'].detach().numpy().flatten()
        tmpop = sequence[ii,1:2]
        if 'opinion_label' in model_output.keys(): 
            test_pred_label = prediction2label(model_output['opinion_label'].detach().numpy())
            tmpop = tmpop/(nclasses-1)
        else: 
            test_pred_label = test_pred * (nclasses-1)

        new_item = np.c_[sequence[ii,:1],test_pred,sequence[ii,2:]]
        train_sequence = np.r_[train_sequence, new_item]

        tmpdf = pd.DataFrame(data = np.c_[sequence[ii,:1], sequence[ii,2:], tmpop, test_pred, test_pred_label], columns=["user","time","gt","pred","pred_label"]) 
        dfs.append(tmpdf)

    res_df = pd.concat(dfs)

    return res_df


def prediction(flag_profile, dataloader, model, batch_size, nclasses):

    model.train = False
    dfs = []
    att_dfs = []
    zu_dfs = []
    for (model_input, gt) in dataloader:
         model_output = model(model_input)
         test_ui = model_input['ui'].numpy().flatten()#[0]
         test_ti = model_input['ti'].numpy().flatten()
         test_oi = gt["opinion"].detach().numpy().flatten()
         test_pred = model_output['opinion'].detach().numpy().flatten()
         if 'opinion_label' in model_output.keys(): 
             test_pred_label = prediction2label(model_output['opinion_label'].detach().numpy())
             test_oi = test_oi/(nclasses-1) 
         else: 
             test_pred_label = test_pred * (nclasses-1)
         tmpdf = pd.DataFrame(data = np.c_[test_ui, test_ti, test_oi, test_pred, test_pred_label], columns=["user","time","gt","pred","pred_label"]) 
         dfs.append(tmpdf)
         if 'zu' in model_output.keys() and not model_output['zu'] is None: 
             zu_pred = model_output['zu'].detach().numpy()
             if test_ui.shape[0]==zu_pred.shape[0]:
                zu_tmpdf = pd.DataFrame(data = np.c_[test_ui[:,np.newaxis], zu_pred], columns=["user"]+list(range(zu_pred.shape[1]))) 
                zu_dfs.append(zu_tmpdf)
         if flag_profile:
             att_pred = model_output['attention'].detach().numpy()[:,:,0]
             att_tmpdf = pd.DataFrame(data = np.c_[test_ui[:,np.newaxis], att_pred], columns=["user"]+list(range(25))) 
             att_dfs.append(att_tmpdf)
    res_df = pd.concat(dfs)
    if flag_profile:
        att_df = pd.concat(att_dfs)
    else:
        att_df = None

    if len(zu_dfs)>0: 
        zu_df = pd.concat(zu_dfs)
    else:
        zu_df = None

    return res_df, att_df, zu_df



def main_sinn(data_type, method, root_path):

    epochs_til_ckpt = 5000 
    steps_til_summary = 5000

    batch_size = opt.batch_size
    hidden_features = opt.hidden_features
    num_hidden_layers = opt.num_hidden_layers
    alpha = opt.alpha
    beta = opt.beta
    num_epochs = opt.num_epochs
    lr = opt.lr
    K = opt.K
    type_odm = opt.type_odm
    activation_func = opt.activation_func
    flag_profile = opt.flag_profile 

    if not "twitter" in data_type: flag_profile = False
    
    ###############################################################################
    # Load data
    ###############################################################################

    print("Loading dataset")
    df = pd.read_csv("working/posts_final_"+data_type+".tsv", delimiter="\t") 
    nclasses = len(df["opinion"].unique())
    if not (method=="SINN" or method=="NN"):
        df["opinion"] = df["opinion"]/(nclasses-1)

    sequence = np.array(df[["user_id","opinion","time"]])
    initial_u = np.loadtxt("working/initial_"+data_type+".txt", delimiter=',', dtype='float')
    
    num_users = int(1 + np.max(sequence[:,0]))
    initial_u = initial_u[:num_users]
    #print("# of users:", num_users, df["user_id"].max())
    #print("# of data points (Positive/Neutral/Negative):", len(df), len(df[df["opinion"]>0.]), len(df[df["opinion"]==0.]), len(df[df["opinion"]<0.]))
    print("Finished loading dataset")

    if flag_profile:
        profiles = np.load("working/hidden_state_profile_"+data_type+".npz")['hidden_state'] 
    else:
        profiles = None

    if "synthetic" in data_type:
        train_period = 0.5
        val_period = 0.7
    else:
        train_period = 0.7
        val_period = 0.8

    train_sequence = sequence[sequence[:,2]<train_period,:]
    train_dataset = load_data(train_sequence, num_users=num_users, initial_u=initial_u)

    val_sequence = sequence[(sequence[:,2]>=train_period)&(sequence[:,2]<val_period),:]
    val_dataset = load_data(val_sequence, num_users=num_users, initial_u=initial_u)

    test_sequence = sequence[(sequence[:,2]>=val_period),:]
    test_dataset = load_data(test_sequence, num_users=num_users, initial_u=initial_u)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=0)


    ###############################################################################
    # Build the model
    ###############################################################################

    if method == 'Voter':
        _method = voter
    elif method == 'DeGroot':
        _method = degroot
    elif method == 'Flocking':
        _method = flocking
    elif method == 'NN':
        _method = nn
    elif method=="AsLM": 
        _method = aslm
    elif method=="SLANT":
        _method = slant
    elif method=="SLANT+":
        _method = slant_plus
    elif method=="SINN":
        _method = sinn

    model = _method.model(type=activation_func, method=type_odm, out_features=num_users, hidden_features=hidden_features, 
                          num_hidden_layers=num_hidden_layers, alpha=alpha, beta=beta, K=K, df_profile=profiles, nclasses=nclasses, dataset=data_type)
    #model.cuda()

    ###############################################################################
    # Training 
    ###############################################################################
    if method!='Voter' and method!='Flocking' and method!='HM':
        training.train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=num_epochs, lr=lr,
                   steps_til_summary=steps_til_summary, epochs_til_checkpoint=epochs_til_ckpt, 
                   model_dir=root_path, loss_fn=_method.loss_function, method=method, input_sequence=sequence)

    model.eval()

    ###############################################################################
    # Evaluation
    ###############################################################################

    if "synthetic" in data_type: 
        for param in model.state_dict().keys(): 
            if "net" in param: continue 
            if "weight" in param: continue 
            #print(param, model.state_dict()[param].numpy()[0])
    
    if type_odm=="External": 
        arr_times = np.linspace(0.,1.,200,endpoint=True)
        X = torch.reshape(torch.tensor(arr_times, dtype=torch.float32), (-1,1))
        At = model.func_A(X).detach().numpy().flatten()
        df_At =  pd.DataFrame({"time": arr_times, "At": At})   
        df_At.to_csv(root_path+"/At_"+method+".csv", index=False)

    if "SLANT" in method or method=="HM" or method=="AsLM": 
        test_res = evaluate(model, sequence, train_sequence, num_users, initial_u, batch_size, nclasses, val_period)
    else:
        test_res, att_res, zu_res = prediction(flag_profile, test_dataloader, model, batch_size, nclasses)
        if not zu_res is None:
            zu_res.to_csv(root_path+"/interaction_predicted_"+method+".csv", index=False)

    test_res.to_csv(root_path+"/test_predicted_"+method+".csv", index=False)
    if flag_profile: 
        att_res.to_csv(root_path+"/attention_predicted_"+method+".csv", index=False)

    train_res, _, _ = prediction(flag_profile, train_dataloader, model, batch_size, nclasses)
    train_res.to_csv(root_path+"/train_predicted_"+method+".csv", index=False)

    val_res, _, _ = prediction(flag_profile, val_dataloader, model, batch_size, nclasses)
    val_res.to_csv(root_path+"/val_predicted_"+method+".csv", index=False)

    mae = (test_res["pred"]-test_res["gt"]).abs()
    print('#######################################')
    print('## Performance for', method, 'on', data_type, 'dataset')
    print("## MAE:", mae.mean())
    if (method=="SINN" or method=="NN" or method=="Voter"):
        truth_label = ((nclasses-1)*test_res["gt"]).astype(np.int)
        acc = accuracy_score(truth_label, test_res["pred_label"])
        f1 = f1_score(truth_label, test_res["pred_label"], average='macro')
        print("## ACC:", acc, "  F1:", f1)
    print('#######################################')
    print()


if __name__ == "__main__":
    logging_root = opt.save_dir

    outdir = os.path.join(logging_root, opt.dataset)
    if not os.path.exists(outdir): os.makedirs(outdir)

    main_sinn(opt.dataset, opt.method, outdir)


