import glob
import math
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
import math
import torch.nn.functional as F
from modules import MLPNet


loss_func = nn.CrossEntropyLoss()


def loss_function(model_output, gt, loss_definition="CE"):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_latent_opinion = gt['opinion']
    pred_latent_opinion = model_output['opinion']


    if loss_definition=="MAE":

        data_losses = (pred_latent_opinion - gt_latent_opinion).pow(2)
        data_loss = data_losses.mean() 

    elif loss_definition=="CE":

        pred_opinion_label = model_output['opinion_label']
        data_loss = loss_func(pred_opinion_label, gt_latent_opinion[:,0].long())


    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss 
           }


class model(MetaModule):

    def __init__(self, out_features=1, type='relu', 
                 method='ODE-DeGroot', hidden_features=256, num_hidden_layers=3, nclasses=None, **kwargs):
        super().__init__()
        self.method = method
        self.out_features = out_features

        profiles = kwargs["df_profile"]
        if profiles is None:
            flag_profile = False
        else:
            flag_profile = True
            profiles = profiles.reshape(-1,25,768)
            self.profiles = torch.from_numpy(profiles.astype(np.float32)).clone() 
        self.flag_profile = flag_profile

        self.net = MLPNet(in_features=2, num_users=out_features, out_features=1, num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, outermost_linear=type, nonlinearity=type, flag_profile=flag_profile)
        self.val2label = nn.Linear(1, nclasses)

        #print(self)

    def forward(self, model_input, params=None):

        times = model_input['ti']
        uids = model_input['ui']

        if self.flag_profile:
            profs = torch.index_select(self.profiles,0,uids[:,0])
            output, attention = self.net(times, uids, profs)
        else:
            output = self.net(times, uids)
            attention = None
        opinion_label = self.val2label(output)

        return {'opinion': output, 'opinion_label': opinion_label, 'attention': attention}

