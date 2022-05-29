import math
import os
import pandas as pd

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import grad

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from collections import OrderedDict
import math
import torch.nn.functional as F
from modules import MLPNet



def gradients_mse(ode_in, ode_out, gradient):
    gradients = diff_gradient(ode_out, ode_in)
    gradients_loss = (gradients - gradient).pow(2).sum(-1)
    return gradients_loss


def diff_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



loss_func = nn.CrossEntropyLoss()

def loss_function(model_output, gt, loss_definition="CE"):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_latent_opinion = gt['opinion']
    pred_latent_opinion = model_output['opinion']
    regularizer_constraint = model_output['regularizer']
    pde_constraints = model_output['pde_constraints']

    pred_opinion_label = model_output['opinion_label']
    data_loss = loss_func(pred_opinion_label, gt_latent_opinion[:,0].long())


    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss, # * 1e2,
            'pde_constraint': pde_constraints.mean(),
            'regularizer_constraint': regularizer_constraint.mean()
           }


def combine(X, Y): 
    X1 = X.unsqueeze(0)
    Y1 = Y.unsqueeze(1)
    X2 = X1.repeat(Y.shape[0],1)
    Y2 = Y1.repeat(1,X.shape[0])
    X3 = torch.reshape(X2, (-1,1))
    Y3 = torch.reshape(Y2, (-1,1))
    return X3, Y3


def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-20
    u = torch.rand(logits.shape) #, minval=0, maxval=1)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)
 

class model(MetaModule):

    def __init__(self, out_features=1, type='relu', 
                 method='DeGroot', hidden_features=256, num_hidden_layers=3, nclasses=None, **kwargs):
        super().__init__()
        self.method = method
        self.out_features = out_features
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.K = kwargs["K"]
        self.dataset = kwargs["dataset"]

        profiles = kwargs["df_profile"]
        if profiles is None:
            flag_profile = False
        else:
            flag_profile = True
            profiles = profiles.reshape(-1,25,768)
            self.profiles = torch.from_numpy(np.array(profiles, dtype=np.float32)).clone()

        self.flag_profile = flag_profile

        self.net = MLPNet(in_features=2, num_users=out_features, out_features=1, num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, outermost_linear=type, nonlinearity=type, flag_profile=flag_profile)
        self.val2label = nn.Linear(1, nclasses)


        if self.method=="DeGroot":
            self.h0 = nn.Parameter(torch.zeros(out_features, self.K))
            self.h1 = nn.Parameter(torch.zeros(out_features, self.K))

        if self.method=="Powerlaw":
            self.mu = nn.Parameter(torch.ones(1))
            self.rho = nn.Parameter(torch.ones(1))

        if self.method=="Mix":
            self.su = nn.Parameter(torch.zeros(out_features)) 

        if self.method=="BCM":
            self.mu = nn.Parameter(torch.ones(1))
            self.consensus_threshold = nn.Parameter(torch.Tensor([1.]))
            self.backfire_threshold = nn.Parameter(torch.Tensor([1.2]))
            self.sigma = nn.Parameter(torch.Tensor([1.]))

        if self.method=="FJ":
            self.su = nn.Parameter(torch.zeros(out_features)) 
        #print(self)


    def sampling(self,vec):
        vec = F.softmax(vec, dim=1)
        logits = gumbel_softmax(vec, 0.1)
        return logits


    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        times = model_input['ti']
        uids = model_input['ui']

        if self.flag_profile:
            profs = torch.index_select(self.profiles,0,uids[:,0])
            output, attention = self.net(times, uids, profs)
        else:
            output = self.net(times, uids)
            attention = None

        users_j = torch.arange(self.out_features)
        if self.training or not "synthetic" in self.dataset or not self.method=="Powerlaw":
            ntau = 1
            taus_j = torch.rand(ntau)  
            users = users_j.unsqueeze(1)
            taus = taus_j.repeat(users.shape[0],1)
            grad_taus_j = taus_j.unsqueeze(1).requires_grad_(True)
        else:
            taus_j = times[:,0]
            ntau = len(taus_j)
            taus, users = combine(taus_j, users_j)
            grad_taus_j = taus.requires_grad_(True) 

        if self.flag_profile:
            _profs = torch.index_select(self.profiles,0,users[:,0])
            _fuv, _ = self.net(taus, users, _profs)
        else:
            _fuv = self.net(taus, users)

        user_id = torch.randint(self.out_features-1, (1,1))
        fuv = torch.transpose(torch.reshape(_fuv, (self.out_features, ntau)), 1, 0)

        if self.training or not "synthetic" in self.dataset or not self.method=="Powerlaw":
            if self.flag_profile:
                _profs = torch.index_select(self.profiles,0,user_id[:,0])
                fu, _ = self.net(grad_taus_j, user_id, _profs)
            else:
                fu = self.net(grad_taus_j, user_id)
        else:
            fu = torch.index_select(fuv, 1, users_j)
        _users = torch.transpose(torch.reshape(users, (self.out_features, ntau)), 1, 0)

        initial_opinion_gt = torch.index_select(model_input['initial'][:1,:], 1, user_id[:,0]) 

        if self.method=="DeGroot":
            h0 = torch.index_select(torch.abs(self.h0),0,user_id[:,0])
            h1 = torch.index_select(torch.abs(self.h1),0,users_j)
            Wuv = torch.matmul(h0, torch.transpose(h1,1,0))
            gt = ( Wuv * fuv ).sum(-1)

            regularizer = self.beta * (torch.norm(h0) + torch.norm(h1))

        if self.method=="BCM":
            mu = torch.abs(self.mu)
            dist = torch.abs(fu - fuv)

            sigma = torch.abs(self.sigma)
            consensus_threshold = torch.abs(self.consensus_threshold)
            consensus_condition = torch.sigmoid( sigma*(consensus_threshold - dist) )
            gt = mu * consensus_condition * (fu - fuv) 
            gt = gt.sum(-1)
            regularizer = self.beta * (torch.norm(sigma) + torch.norm(mu)) 


        consensus_z = None 
        if self.method=="Powerlaw":
            mu = torch.abs(self.mu)
            dist = torch.abs(fu - fuv)
            pij = (dist + 1e-12).pow(self.rho)
            consensus_z = self.sampling(pij)
            gt = consensus_z * (fu - fuv)
            gt = gt.sum(-1)

            regularizer = self.beta * torch.norm(mu)

        if self.method=="Mix":
            dist = torch.abs(fu - fuv)
            su = torch.gather(self.su,0,user_id[:,0])
            gt = torch.abs(su) * fuv.sum(-1)
            regularizer = self.beta * (torch.norm(su))

        if self.method=="FJ":
            su = torch.gather(self.su,0,user_id[:,0])
            su = torch.abs(su)
            gt = su * fuv.sum(-1) + (1.-su) * initial_opinion_gt - fu 
            regularizer = self.beta * (torch.norm(su))


        gt = torch.reshape(gt, (-1,ntau))

        pde_constraints = gradients_mse(grad_taus_j, fu, gt)
        pde_constraint = self.alpha * pde_constraints

        opinion_label = self.val2label(output)

        return {'opinion': output, 'opinion_label': opinion_label, 'pde_constraints': pde_constraints, 
                'regularizer': regularizer, 'attention': attention, 'zu': consensus_z}

