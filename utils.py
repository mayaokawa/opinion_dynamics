import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision.utils import make_grid, save_image
#import skimage.measure
import cv2
#import meta_modules
import scipy.io.wavfile as wavfile
import seaborn as sns

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
sns.set_palette('tab10')

T = 100
N = 25

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_opinion_summary(model, model_input, gt, model_output, writer, total_steps, out_dir, method, sequence=None, xvals=None, yvals=None):

    uids = sequence[:, :1]
    latent_opinion = sequence[:, 1:2]
    times = sequence[:, 2:3]

    tensor_times = torch.Tensor(times)
    tensor_uids = torch.Tensor(uids).long()


    plt.figure(figsize=(7,3.5))
    for tmpuid in range(N): 
        tmptime = times[uids==tmpuid]
        tmpod = latent_opinion[uids==tmpuid]
        plt.plot(tmptime+0.5, tmpod)
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.ylim(-1,1)
    plt.tight_layout()
    plt.savefig(out_dir+"/observed.png"), plt.close()


    slice_time = np.linspace(-0.5,0.5,T+1).reshape(-1,1)
    slice_times = np.repeat(slice_time, N, axis=1).reshape(-1,1)
    slice_uid = np.arange(N).reshape(1,-1)
    slice_uids = np.repeat(slice_uid, T+1, axis=0).reshape(-1,1)

    tensor_slice_times = torch.Tensor(slice_times)
    tensor_slice_uids = torch.Tensor(slice_uids).long()

    with torch.no_grad():
        yz_slice_model_input = {'times': tensor_slice_times[None, ...], 'uids': tensor_slice_uids[None, ...]}
        yz_model_out = model(yz_slice_model_input)
        yz_pred = yz_model_out['model_out'].numpy()[0,:,:]

        plt.figure(figsize=(7,3.5))
        for tmpuid in range(N): 
            tmptime = slice_times[slice_uids==tmpuid]
            tmpod = yz_pred[slice_uids==tmpuid]
            plt.plot(tmptime+0.5, tmpod)
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        plt.ylim(-1,1)
        plt.tight_layout()
        plt.savefig(out_dir+"/estimated_"+method+".png"), plt.close()


        test_model_input = {'times': tensor_times[None, ...], 'uids': tensor_uids[None, ...]}
        test_model_out = model(test_model_input)
        test_pred = test_model_out['model_out'].numpy()[0,:,:]

        res = np.c_[times,uids,latent_opinion,test_pred]
        np.savetxt(out_dir+"/accuracy_"+method+".txt", res, delimiter=",", fmt="%.4f")



