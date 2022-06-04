import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import grad

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from modules import MLPNet
import torch.nn.functional as F


def gradients_mse(ode_in, ode_out, rhs):
    gradients = diff_gradient(ode_out, ode_in)  ## Left hand side of ODE $\tilde{x}_u(t)/dt$ 
    ODE_loss = (gradients - rhs).pow(2).sum(-1)
    return ODE_loss


def diff_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



loss_func = nn.CrossEntropyLoss()

def loss_function(model_output, gt, loss_definition="CE"):

    
    pred_opinion_label = model_output['opinion_label'] ## Predicted opinion label
    gt_latent_opinion = gt['opinion'] ## Ground truth opinion label

    ### Compute data loss $\mathcal{L}_{data}$
    data_loss = loss_func(pred_opinion_label, gt_latent_opinion[:,0].long())


    regularizer = model_output['regularizer']
    ode_constraint = model_output['ode_constraint']

    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss, # * 1e2,
            'ode_constraint': ode_constraint.mean(),
            'regularizer_constraint': regularizer.mean()
           }



def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-20
    u = torch.rand(logits.shape)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)
 

class model(MetaModule):

    def __init__(self, num_users=1, type='relu',  
                 hidden_features=256, num_hidden_layers=3, nclasses=None, **kwargs):
        super().__init__()

        self.U = num_users  ## Number of users
        self.type_odm = kwargs["type_odm"]  ## Choice of opinion dynamics model 

        ### Set hyperparameters
        self.alpha = kwargs["alpha"]  ## Trade-off hyperparameter $\alpha$
        self.beta = kwargs["beta"]  ## Regularization parameter $\beta$
        self.K = kwargs["K"]  ## Dimension of the latent space 
        self.J = 1  ## Number of collocation points $J$

        ### Prepare user profile information
        profiles = kwargs["df_profile"]  ## Hidden user representation $\{{\bf h}_1,...,{\bf h}_U\}$
        if profiles is None:
            use_profile = False
        else:
            use_profile = True
            profiles = profiles.reshape(-1,25,768)
            self.profiles = torch.from_numpy(np.array(profiles, dtype=np.float32)).clone()
        self.use_profile = use_profile

        ### Prepare neural network $f(t,\{bf e\}_u;\theta_f)$
        self.net = MLPNet(num_users=self.U, num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, outermost_linear=type, nonlinearity=type, use_profile=use_profile)
        self.val2label = nn.Linear(1, nclasses)


        ### Prepare ODE parameters $\Lambda$ 
        if self.type_odm=="DeGroot":
            self.M = nn.Parameter(torch.rand(self.U, self.K)/self.U) ## Latent matrix $M\in\mathbb{R}^{U \times K}$
            self.Q = nn.Parameter(torch.rand(self.U, self.K)/self.U) ## Latent matrix $Q\in\mathbb{R}^{U \times K}$

        if self.type_odm=="SBCM":
            self.rho = nn.Parameter(torch.ones(1))  ## Exponent parameter $\rho$ 

        if self.type_odm=="BCM":
            self.mu = nn.Parameter(torch.ones(1))
            self.threshold = nn.Parameter(torch.Tensor([1.]))  ## Confidence threshold $\delta$ 
            self.sigma = nn.Parameter(torch.Tensor([1.]))

        if self.type_odm=="FJ":
            self.s_u = nn.Parameter(torch.zeros(self.U))  ## Users' susceptibilities $\{s_1,..,s_U\}$
        #print(self)


    def sampling(self,vec):
        vec = F.softmax(vec, dim=1)
        logits = gumbel_softmax(vec, 0.1)
        return logits


    def forward(self, model_input, params=None):

        ### Set neural network input: times $t$ and user id $u$
        times = model_input['ti']
        uids = model_input['ui']

        ### Get neural network output \hat{x}_u(t)
        if self.use_profile:
            profs = torch.index_select(self.profiles,0,uids[:,0])
            output, attention = self.net(times, uids, profs)
        else:
            output = self.net(times, uids)
            attention = None

        ### Predict opinion labels
        opinion_label = self.val2label(output)


        ### Setup ODE constraints
        tilde_z_ut = None 

        if self.training:
            ### Sample $J$ collocation points $\{\tau_1,...,\tau_J\}$ from time domain $\tau_j\in[0,T+\Delta T]$ with $T+\Delta T = 1$
            tau_j = torch.rand(self.J).unsqueeze(1).requires_grad_(True)  

            users = torch.arange(self.U).unsqueeze(1)
            taus = tau_j.repeat(users.shape[0],1)

            if self.use_profile:
                _profs = torch.index_select(self.profiles,0,users[:,0])
                _vector_x, _ = self.net(taus, users, _profs)
            else:
                _vector_x = self.net(taus, users)

            ## Predicted opinions of $U$ users $\{\tilde{x}_1(\tau_j),...,\tilde{x}_U(\tau_j)\}$
            vector_x = torch.transpose(torch.reshape(_vector_x, (self.U, self.J)), 1, 0)


            user_id = torch.randint(self.U-1, (1,1))  ### Sample user $u$
            if self.use_profile:
                _profs = torch.index_select(self.profiles,0,user_id[:,0])
                x_u, _ = self.net(tau_j, user_id, _profs)
            else:
                ### Predict opinion $\tilde{x}_u(\tau_j)$ of user $u$ at time $\tau_j$
                x_u = self.net(tau_j, user_id)  


            if self.type_odm=="DeGroot":
                m_u = torch.index_select(torch.abs(self.M),0,user_id[:,0])
                Q = torch.abs(self.Q)
                a_u = torch.matmul(m_u, torch.transpose(Q,1,0))

                ## Right hand side (rhs) of Equation (5)
                rhs_ode = torch.matmul(a_u, vector_x.T)

                ## Regularization term $\mathcal{R}(\Lambda)$
                regularizer = self.beta * (torch.norm(self.M) + torch.norm(self.Q)) 

            if self.type_odm=="SBCM":
                distance = torch.abs(x_u - vector_x)

                ## Probability of user $u$ selecting user $v$ as an interaction partner at time $\tau_j$
                p_uv = (distance + 1e-12).pow(self.rho)

                ## Differentiable one-hot approximation $\tilde{z}_u^t$ in Equation (9)
                tilde_z_ut = self.sampling(p_uv)

                ## Right hand side (rhs) of Equation (10)
                rhs_ode = tilde_z_ut * (x_u - vector_x)
                rhs_ode = rhs_ode.sum(-1)

                ## Regularization term $\mathcal{R}(\Lambda)$
                regularizer = self.beta * torch.zeros(1)

            if self.type_odm=="BCM":
                mu = torch.abs(self.mu)
                distance = torch.abs(x_u - vector_x)

                ## Prepare ODE parameters 
                sigma = torch.abs(self.sigma)
                threshold = torch.abs(self.threshold)

                ## Right hand side (rhs) of Equation (8)
                rhs_ode = mu * torch.sigmoid( sigma*(threshold - distance) ) * (x_u - vector_x) 
                rhs_ode = rhs_ode.sum(-1)

                ## Regularization term $\mathcal{R}(\Lambda)$
                regularizer = self.beta * (torch.norm(sigma) + torch.norm(mu)) 

            if self.type_odm=="FJ":
                ## Initial opinions of $U$ users 
                initial_opinion_gt = torch.index_select(model_input['initial'][:1,:], 1, user_id[:,0]) 

                ## Get user $u$'s susceptibility to persuasion
                s_u = torch.gather(torch.abs(self.s_u),0,user_id[:,0])  

                ## Right hand side (rhs) of Equation (7)
                rhs_ode = s_u * vector_x.sum(-1) + (1.-s_u) * initial_opinion_gt - x_u

                ## Regularization term $\mathcal{R}(\Lambda)$
                regularizer = self.beta * torch.norm(s_u)

            rhs_ode = torch.reshape(rhs_ode, (-1,self.J))

            ### Compute ODE loss $\mathcal{L}_{ode}$
            ode_constraints = gradients_mse(tau_j, x_u, rhs_ode)
            ode_constraint = self.alpha * ode_constraints
        else:
            ode_constraint = torch.zeros(1)
            regularizer = torch.zeros(1)


        return {'opinion': output, 'opinion_label': opinion_label, 'ode_constraint': ode_constraint, 
                'regularizer': regularizer, 'attention': attention, 'zu': tilde_z_ut}

