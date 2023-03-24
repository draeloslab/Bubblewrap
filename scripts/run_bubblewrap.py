import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

from bubblewrap import Bubblewrap

from math import atan2, floor
from tqdm import tqdm
import pdb


# ## Parameters
# N = 1000             # number of nodes to tile with
# lam = 1e-3          # lambda 
# nu = 1e-3           # nu
# eps = 1e-3          # epsilon sets data forgetting
# step = 8e-2         # for adam gradients
# M = 30              # small set of data seen for initialization
# B_thresh = -10      # threshold for when to teleport (log scale)    
# batch = False       # run in batch mode 
# batch_size = 1      # batch mode size; if not batch is 1
# go_fast = False     # flag to skip computing priors, predictions, and entropy for optimal speed

default_parameters = dict(
N = 100,       
lam = 1e-3,
nu = 1e-3,      
eps = 1e-3,     
step = 8e-2,    
M = 30,         
B_thresh = -10, 
batch = False,  
batch_size = 1, 
go_fast = False,
)

# default_file = "generated/vdp_1trajectories_2dim_500to20500_noise0.05.npz"
default_file = "generated/vdp_1trajectories_2dim_500to20500_noise0.2.npz"
# default_file = "generated/lorenz_1trajectories_3dim_500to20500_noise0.05.npz"

def run_bubblewrap(file, params):
    s = np.load(file)
    data = s['y'][0]

    T = data.shape[0]       # should be 20k
    d = data.shape[1]       # should be 2


    bw = Bubblewrap(params["N"], 
                    d, 
                    step=params["step"], 
                    lam=params["lam"], 
                    M=params["M"], 
                    eps=params["eps"], 
                    nu=params["nu"], 
                    B_thresh=params["B_thresh"], 
                    batch=params["batch"], 
                    batch_size=params["batch_size"], 
                    go_fast=params["go_fast"]
                    ) 

    ## Set up for online run through dataset
    init = -params["M"]
    end = T-params["M"]
    step = params["batch_size"]

    ## Initialize things
    for i in np.arange(0, params["M"], step): 
        if params["batch"]:
            bw.observe(data[i:i+step]) 
        else:
            bw.observe(data[i])
    bw.init_nodes()

    ## Run online, 1 data or batch at a time
    for i in tqdm(np.arange(init, end, step)):
        bw.observe(data[i+params["M"]:i+params["M"]+step])
        bw.e_step()  
        bw.grad_Q()
    return bw



def plot_bubblewrap_results(bw):
    T = len(np.array(bw.pred_far)) # TODO: confirm that this T calculation is correct

    plt.figure()
    plt.plot(bw.pred_far)
    print('Mean pred ahead: ', np.mean(np.array(bw.pred_far)[-floor(T/2):]))
    var_tmp = np.convolve(bw.pred_far, np.ones(500)/500, mode='valid')
    plt.plot(var_tmp, 'k')


    plt.figure()
    plt.plot(bw.entropy_list)
    print('Mean entropy: ', np.mean(np.array(bw.entropy_list)[-floor(T/2):]))
    var_tmp = np.convolve(bw.entropy_list, np.ones(500)/500, mode='valid')
    plt.plot(var_tmp, 'k')
    plt.show()

def save_data_for_later_plotting(bw,file):
    # file is the original file that bubblewrap ran on
    if "vdp" in file:
        prefix = "vdp_2d"
    else:
        prefix = "lorenz_3d"
    A = np.save(f"generated/{prefix}_A.npy", bw.A)
    mu = np.save(f"generated/{prefix}_mu.npy", bw.mu)
    L = np.save(f"generated/{prefix}_L.npy", bw.L)
    n_obs = np.save(f"generated/{prefix}_n_obs.npy", bw.n_obs)
    pred = np.save(f"generated/{prefix}_pred.npy", bw.pred)
    entropy = np.save(f"generated/{prefix}_entropy.npy", bw.entropy_list)



if __name__ == "__main__":
    bw = run_bubblewrap(default_file, default_parameters)
    pdb.set_trace()

    plot_bubblewrap_results(bw)
    save_data_for_later_plotting(bw, default_file)



