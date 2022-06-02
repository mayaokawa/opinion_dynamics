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

        self.nu = nn.Parameter(0.05*torch.rand(1,1))
        self.w = nn.Parameter(0.05*torch.rand(1,1))
        self.alpha = nn.Parameter(torch.rand(num_users))
        self.beta = nn.Parameter(torch.rand(num_users))
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=1, 
                            batch_first=True,
                            bidirectional=False)

    def forward(self, model_input, params=None):

        history = model_input['history']
        uj = history[:,:,0].long()
        mj = history[:,:,1]
        tj = history[:,:,2]
        ui = model_input['ui']
        ti = model_input['ti']

        delta_ij = ti - tj[:,-2:-1]
        _delta_ij = tj[:,-1:] - tj[:,-2:-1]
        delta_m = mj[:,-1:] - mj[:,:-1]
        alphai = torch.index_select(self.alpha,0,ui[:,0]).unsqueeze(-1)
        betai = torch.index_select(self.beta,0,ui[:,0]).unsqueeze(-1)

        users_j = torch.arange(self.num_users)

        A0 = self.A0(ui[:,0]).abs()
        A1 = self.A1(users_j).abs()
        Ai = torch.matmul(A0, torch.transpose(A1, 1, 0))
        _A1 = self.A1(uj[:,-2:-1]).abs()
        Aij = torch.einsum('ij,ikj->ik', A0, _A1)

        B0 = self.B0(ui[:,0]).abs()
        B1 = self.B1(users_j).abs()
        Bi = torch.matmul(B0, torch.transpose(B1, 1, 0)) 
        _B1 = self.B1(uj[:,-2:-1]).abs()
        Bij = torch.einsum('ij,ikj->ik', B0, _B1)

        C0 = self.C0(ui[:,0]).abs()
        C1 = self.C1(users_j).abs()
        Ci = torch.matmul(C0, torch.transpose(C1, 1, 0))
        _C1 = self.C1(uj[:,:-1]).abs()
        Cij = torch.einsum('ij,ikj->ik', C0, _C1)

        zij = self.sigmoid(Cij * torch.abs(delta_m))
        zij = torch.mean(zij, axis=-1).unsqueeze(-1)

        lstm_input = mj[:,-2:-1] * Aij + tj[:,-2:-1] * Bij + zij 
        lstm_input = lstm_input.unsqueeze(-1)
        hij, _ = self.lstm(lstm_input)

        w = torch.abs(self.w)
        nu = self.nu

        xi = 0.1 * w * torch.exp( hij[:,:,0] + w * delta_ij + alphai )
        lamdai = nu * torch.exp( hij[:,:,0] + nu * delta_ij + betai )

        log_l = torch.log(lamdai)
        Int_l = torch.exp( hij[:,:,0] + nu * _delta_ij + betai ) - torch.exp( hij[:,:,0] + nu * delta_ij + betai )

        return {'log_l': log_l, 'Int_l': Int_l, 'opinion': xi}


    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        times = model_input['times'].clone().detach().requires_grad_(True)
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


