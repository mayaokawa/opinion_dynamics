import csv
import glob
import math
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
import math
import torch.nn.functional as F


class model(MetaModule):

    def __init__(self, out_features=1, type='relu', method=None,
                 hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()

        A = torch.ones(out_features,out_features)
        self.A = A / out_features 
        self.out_features = out_features
        #print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        previous = model_input['previous']
        uids = model_input['ui']

        uid = torch.randint(self.out_features-1, uids.shape)
        output = torch.gather(previous,1,uid)
        
        return {'opinion': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        times = model_input['times'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(times)
        return {'opinion': activations.popitem(), 'activations': activations}



def loss_function(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_latent_opinion = gt['opinion']
    pred_latent_opinion = model_output['opinion']

    data_loss = (pred_latent_opinion - gt_latent_opinion)**2
    #print(data_loss.mean())

    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss.mean(), 
           }



