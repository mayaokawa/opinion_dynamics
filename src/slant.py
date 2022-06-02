import numpy as np
import torch
from torch.utils.data import Dataset

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)


class model(MetaModule):

    def __init__(self, num_users=1, type='relu', 
                 hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()

        self.A0 = nn.Embedding(num_users, 1)
        self.A1 = nn.Embedding(num_users, 1)
        self.B0 = nn.Embedding(num_users, 1)
        self.B1 = nn.Embedding(num_users, 1)
        self.C0 = nn.Embedding(num_users, 1)
        self.C1 = nn.Embedding(num_users, 1)
        self.num_users = num_users
        self.nu = nn.Parameter(0.05*torch.rand(1,1,1))
        self.w = nn.Parameter(0.05*torch.rand(1,1,1))
        self.alpha = nn.Parameter(torch.rand(num_users))
        self.beta = nn.Parameter(torch.rand(num_users))
        self.lstm = nn.LSTM(input_size=1, #config.emb_dim + 1,
                            hidden_size=hidden_features,
                            batch_first=True,
                            bidirectional=False)

    def forward(self, model_input, params=None):

        history = model_input['history']
        uj = history[:,:,0].long()
        mj = history[:,:,1]
        tj = history[:,:,2]
        ui = model_input['ui']
        ti = model_input['ti']

        delta_ij = ti - tj[:,:-1]
        _delta_ij = tj[:,-1:] - tj[:,:-1]

        alpha = torch.abs(self.alpha)
        beta = torch.abs(self.beta)
        alphai = torch.index_select(alpha,0,ui[:,0])
        betai = torch.index_select(beta,0,ui[:,0])

        users_j = torch.arange(self.num_users)

        A0 = self.A0(ui[:,0]).abs()
        A1 = self.A1(users_j).abs()
        Ai = torch.matmul(A0, torch.transpose(A1, 1, 0)) #.unsqueeze(0)
        _A1 = self.A1(uj[:,-2:-1]).abs()
        Aij = torch.einsum('ij,ikj->ik', A0, _A1)

        B0 = self.B0(ui[:,0]).abs()
        B1 = self.B1(users_j).abs()
        Bi = torch.matmul(B0, torch.transpose(B1, 1, 0)) #.unsqueeze(0)
        _B1 = self.B1(uj[:,-2:-1]).abs()
        Bij = torch.einsum('ij,ikj->ik', B0, _B1)

        w = torch.abs(self.w)
        xi = alphai + torch.mean(mj[:,:-1] * Aij * w * torch.exp(-w * delta_ij), axis=-1)
        xi = torch.unsqueeze(xi,-1)
        nu = self.nu
        lamdai = betai + torch.mean(Bij * nu * torch.exp(-nu * delta_ij), axis=-1)

        log_l = torch.log(lamdai)
        Int_l = torch.sum( torch.unsqueeze(alphai,-1) * delta_ij + (torch.exp(-nu * _delta_ij) - torch.exp(-nu * delta_ij)), axis=-1) 

        return {'log_l': log_l, 'Int_l': Int_l, 'opinion': xi}


    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        times = model_input['times'] #.clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(times)
        return {'model_out': activations.popitem(), 'activations': activations}


def loss_function(model_output, gt, loss_definition="MAE"):
    gt_latent_opinion = gt['opinion']
    pred_latent_opinion = model_output['opinion']
    logistic_loss = 1. / (1. + torch.exp(-1. * gt_latent_opinion*pred_latent_opinion)) 
    data_loss = model_output['Int_l'] - model_output['log_l'] 


    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss.mean(), 
            'logistic_loss': logistic_loss.mean()
           }


