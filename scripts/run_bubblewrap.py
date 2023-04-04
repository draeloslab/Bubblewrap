import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import datetime

from bubblewrap import Bubblewrap

from math import atan2, floor
from tqdm import tqdm


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
num = 100,
lam = 1e-3,
nu = 1e-3,
eps = 1e-3,
step = 8e-2,
M = 30,
B_thresh = -10,
batch = False,
batch_size = 1,
go_fast = False,
lookahead_steps = [1,10]
)
# default_file = "generated/vdp_1trajectories_2dim_500to20500_noise0.05.npz"
default_file = "generated/vdp_1trajectories_2dim_500to20500_noise0.2.npz"
# default_file = "generated/lorenz_1trajectories_3dim_500to20500_noise0.05.npz"

def run_bubblewrap(file, params):
    s = np.load(file)
    data = s['y'][0]

    T = data.shape[0]       # should be 20k
    d = data.shape[1]       # should be 2


    bw = Bubblewrap(d, **params)

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



def plot_bubblewrap_results(bw, running_average_length=500):
    T = len(bw.pred_list)

    pred_mat = np.array(bw.pred_list)
    ent_mat = np.array(bw.entropy_list)


    for step_n in range(len(bw.lookahead_steps)):
        steps = bw.lookahead_steps[step_n]
        fig, ax = plt.subplots(1, 2, sharex='all')
        pred = pred_mat[:, step_n]
        ax[0].plot(pred)
        print(f'Mean pred ahead ({steps} steps): {np.mean(pred[-floor(T/2):])}')

        var_tmp = np.convolve(pred, np.ones(running_average_length)/running_average_length, mode='valid')
        var_tmp_x = np.arange(var_tmp.size) + running_average_length//2
        ax[0].plot(var_tmp_x, var_tmp, 'k')
        ax[0].set_title(f"Prediction ({steps} steps)")

        ent = ent_mat[:, step_n]
        ax[1].plot(ent)
        print(f'Mean entropy ({steps} steps): {np.mean(ent[-floor(T/2):])}')
        var_tmp = np.convolve(ent, np.ones(running_average_length)/running_average_length, mode='valid')
        var_tmp_x = np.arange(var_tmp.size) + running_average_length//2
        ax[1].plot(var_tmp_x, var_tmp, 'k')
        ax[1].set_title(f"Entropy ({steps} steps)")
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
    pred = np.save(f"generated/{prefix}_pred.npy", bw.pred_list)
    entropy = np.save(f"generated/{prefix}_entropy.npy", bw.entropy_list)

class BubblewrapRun:
    # TODO: if we go forward with this, I need to clean up the previous plotting/saving stuff
    def __init__(self, bw: Bubblewrap, file, bw_parameters=None):
        self.file = file
        self.bw_parameters = bw_parameters

        self.A = bw.A
        self.mu = bw.mu
        self.L = bw.L
        self.n_obs = bw.n_obs
        self.pred_list = np.array(bw.pred_list)
        self.entropy_list = np.array(bw.entropy_list)
        self.dead_nodes = bw.dead_nodes

    def save(self, dir="generated"):
        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        with open(os.path.join(dir, f"bubblewrap_run_{time_string}.pickle"), "wb") as fhan:
            pickle.dump(self, fhan)


if __name__ == "__main__":
    bw = run_bubblewrap(default_file, default_parameters)

    plot_bubblewrap_results(bw)
    save_data_for_later_plotting(bw, default_file)

    br = BubblewrapRun(bw,file=default_file, bw_parameters=default_parameters)
    br.save()