import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F



class MLPNet(nn.Module):
    def __init__(self, num_users, num_hidden_layers, hidden_features, out_features=1,
                 outermost_linear='sigmoid', nonlinearity='relu', use_profile=False):
        super(MLPNet, self).__init__()

        nls = {'relu': nn.ReLU(inplace=True), 
               'sigmoid': nn.Sigmoid(), 
               'tanh': nn.Tanh(), 
               'selu': nn.SELU(inplace=True), 
               'softplus': nn.Softplus(), 
               'elu': nn.ELU(inplace=True)}

        nl = nls[nonlinearity]
        nl_outermost = nls[outermost_linear]

        self.use_profile = use_profile
        if use_profile:
            self.embed_profiles = [] 
            self.embed_profiles.append(nn.Sequential(
                nn.Linear(768, hidden_features), nl
            ))
            self.embed_profiles = nn.Sequential(*self.embed_profiles)

            self.embed_att = [] 
            self.embed_att.append(nn.Sequential(
                nn.Linear(hidden_features, 1), nn.Softmax()
            ))
            self.embed_att = nn.Sequential(*self.embed_att)

        self.hidden_features = hidden_features

        self.embed_users = [] 
        self.embed_users.append(nn.Sequential(
                nn.Embedding(num_users, hidden_features), nl
        ))
        for i in range(num_hidden_layers-1):
            self.embed_users.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_users = nn.Sequential(*self.embed_users)

        self.embed_times = []
        self.embed_times.append(nn.Sequential(
            nn.Linear(1, hidden_features), nl
        ))
        for i in range(num_hidden_layers-1):
            self.embed_times.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_times = nn.Sequential(*self.embed_times)

        self.net = []
        if use_profile:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features*3, hidden_features), nl
            ))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features*2, hidden_features), nl
            ))
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nl_outermost
        ))
        self.net = nn.Sequential(*self.net)

        
    def forward(self, times, users, profs=None, params=None, **kwargs):

        x = self.embed_times(times.float())
        y = self.embed_users(users.long())
        y = torch.squeeze(y, dim=1)

        if self.use_profile: 
            z = self.embed_profiles(profs.float())
            att = self.embed_att(z)
            z = torch.mean(att*z, axis=1)
            combined = torch.cat([x, y, z], dim=-1)
        else: 
            combined = torch.cat([x, y], dim=-1)
        output = self.net(combined)

        if self.use_profile: 
            return output, att
        else:
            return output


