import os
import numpy as np
import torch
from torch.utils.data import Dataset

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)


class model(MetaModule):

    def __init__(self, num_users=1, type='relu', 
                 hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()

        self.num_users = num_users
        self.beta = kwargs["beta"]

        self.W = nn.Parameter(torch.rand(num_users,num_users))
        #print(self)

    def forward(self, model_input, params=None):

        previous = model_input['previous']
        uids = model_input['ui']


        users_j = torch.arange(self.num_users)
        W = torch.abs(self.W)
        Wu = torch.index_select(W,0,uids[:,0])
        Wuv = torch.index_select(Wu,1,users_j)

        output = torch.sum(Wuv * previous, axis=-1)
        output = torch.unsqueeze(output, 1)
        constraints = self.beta * torch.sum(W) 
        
        return {'opinion': output, 'constraints': constraints}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        times = model_input['times'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(times)
        return {'opinion': activations.popitem(), 'activations': activations}



def loss_function(model_output, gt, loss_definition="MAE"):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_latent_opinion = gt['opinion']
    pred_latent_opinion = model_output['opinion']
    pde_constraint = model_output['constraints'] 

    data_loss = (pred_latent_opinion - gt_latent_opinion)**2

    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss.mean(), 
            'pde_constraint': pde_constraint.mean()
           }



