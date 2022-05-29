import numpy as np
#import sys
#from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import bernoulli


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def generate(N, Q, T, interactions_per_timestep=10,
             actions_per_timestep_per_node=10,
             mu=0.1, exponents=0.6, 
             feature_sharpness=8, fixed_sigma=0.5, seed=None):

    x0 = np.random.uniform(size=N) * 2 - 1

    u_v_t_w = []

    t = 0
    X = [x0]

    for t in range(T):
        xt = X[-1]


        xtp1 = xt.copy()
        for _ in range(interactions_per_timestep):
            i = np.random.randint(N)

            if exponents==0.0:
               j = np.random.choice(N)
               dist = np.abs(xt[j] - xt[i])
               if dist < 0.01:
                  xtp1[i] += mu * xt[j]
            else:
               dist = np.abs(xt - xt[i])
               pij = np.power(dist + 1e-6, -exponents) / np.sum(np.power(dist + 1e-6, -exponents))
               j = np.random.choice(N, p=pij)
               xtp1[i] += mu * xt[j]
               u_v_t_w.append( np.array([i, j, t]) )

            xtp1 = np.clip(xtp1, -1, 1)
            
        X.append(xtp1)

    X = np.vstack(X) 
    u_v_t_w = np.vstack(u_v_t_w) 
    X = X 

    return X, u_v_t_w

hyperparams_settings = {
    'consensus': {'mu': 0.05, 'exponents': -1.0},
    'clustering': {'mu': 0.05, 'exponents': 0.05},
    'polarization': {'mu': 0.02, 'exponents': 0.5},
}


def main(
    N=200, # Number of users 
    Q=20, 
    T=200, # Number of time steps
    actions_per_timestep_per_node=1,
    interactions_per_timestep=50, #90,
    num_generations=1,
    fixed_sigma=0.1,
):
    for experiment_run_id in range(num_generations):
        for generative_id, generative_setting in hyperparams_settings.items():
            generative_seed = hash(f"{experiment_run_id}-{generative_id}") & (2**32 - 1)
            X_original, u_v_t_w = generate(N=N, Q=Q, T=T,
                                  actions_per_timestep_per_node=actions_per_timestep_per_node,
                                  fixed_sigma=fixed_sigma,
                                  interactions_per_timestep=interactions_per_timestep,
                                  seed=generative_seed,
                                  **generative_setting)

            X_original = X_original 
            time = np.linspace(0,1,T+1).reshape(-1,1)
            times = np.repeat(time, N, axis=1)
            uid = np.arange(N).reshape(1,-1)
            uids = np.repeat(uid, T+1, axis=0)
            data = np.c_[uids.reshape(-1,1),X_original.reshape(-1,1),times.reshape(-1,1)]
            np.savetxt("working/synthetic_"+generative_id+".csv", data, delimiter=",", fmt="%.4f")
            np.savetxt("working/synthetic_interaction_"+generative_id+".csv", u_v_t_w, delimiter=",", fmt="%.4f")



if __name__ == '__main__':
    main()

