import numpy as np
import torch
from torch.utils.data import Dataset

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from modules import MLPNet


loss_func = nn.CrossEntropyLoss()


def loss_function(model_output, gt):

    gt_latent_opinion = gt['opinion']
    pred_latent_opinion = model_output['opinion']

    pred_opinion_label = model_output['opinion_label']
    data_loss = loss_func(pred_opinion_label, gt_latent_opinion[:,0].long())


    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss 
           }


class model(MetaModule):

    def __init__(self, num_users=1, type='relu', 
                 hidden_features=256, num_hidden_layers=3, nclasses=None, **kwargs):
        super().__init__()
        self.num_users = num_users

        profiles = kwargs["df_profile"]
        if profiles is None:
            flag_profile = False
        else:
            flag_profile = True
            profiles = profiles.reshape(-1,25,768)
            self.profiles = torch.from_numpy(profiles.astype(np.float32)).clone() 
        self.flag_profile = flag_profile

        self.net = MLPNet(num_users=num_users, num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, outermost_linear=type, nonlinearity=type)
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

