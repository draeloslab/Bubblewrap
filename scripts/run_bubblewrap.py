import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import numpy as np

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pylab as plt
from matplotlib.animation import FFMpegFileWriter

from bubblewrap import Bubblewrap
from plot_2d_3d import plot_2d
from bubblewrap_run import BubblewrapRun

from math import atan2, floor
from tqdm import tqdm

# todo: unify movie and non-movie functions

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
    num=200,
    lam=1e-3,
    nu=1e-3,
    eps=1e-3,
    step=8e-2,
    M=30,
    B_thresh=-10,
    batch=False,
    batch_size=1,
    go_fast=False,
    lookahead_steps=[1, 2, 4 ,8, 16, 32, 64, 128, 256, 512, 1024],
    seed=42,
)


def run_bubblewrap(file, params, keep_every_nth_frame=None, do_it_old_way=False, end=None):
    """this runs bubblewrap; it also generates a movie if `keep_every_nth_frame` is not None"""
    if keep_every_nth_frame is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10,5), squeeze=True)

        moviewriter = FFMpegFileWriter(fps=20)
        moviewriter.setup(fig, "generated/movie.mp4", dpi=100)
    else:
        moviewriter = None

    s = np.load(file)

    # todo: give all files uniform format
    if "neuropixel" in file:
        data = s['ssSVD10'].T
        # data = s['ssSVD20'].T
    elif ("jpca" in file) or ("mouse" in file) or ("widefield" in file):
        data = s.T
    else:
        data = s['y'][0]

    T = data.shape[0]       # should be big (like 20k)
    d = data.shape[1]       # should be small-ish (like 2)
    bw = Bubblewrap(d, **params)

    ## Set up for online run through dataset

    init = -params["M"]
    if end is None:
        # end = T-(params["M"] + max(bw.lookahead_steps))
        end = T - params["M"]
    step = params["batch_size"]

    # Initialize things
    for i in np.arange(0, params["M"], step):
        if params["batch"]:
            bw.observe(data[i:i+step])
        else:
            bw.observe(data[i])
    bw.init_nodes()

    # Run online, 1 data or batch at a time
    for i in tqdm(np.arange(init, end, step)):
        start_of_block = i+params["M"]
        end_of_block = i+params["M"]+step
        bw.observe(data[start_of_block:end_of_block])

        future_observations = {}
        for x in bw.lookahead_steps:
            if (end_of_block - 1) + (x - 1) < T:
                future_observations[x] = data[(end_of_block - 1) + (x - 1)]

        bw.e_step(future_observations, do_it_old_way=do_it_old_way)
        bw.grad_Q()

        if keep_every_nth_frame is not None:
            assert step == 1 # the keep_every_th assumes the step is 1
            if i % keep_every_nth_frame == 0:
                ax[0].cla()


                d = data[0:i+params["M"]+step]
                plot_2d(ax[0], d, bw.A, bw.mu, bw.L, np.array(bw.n_obs), bw)

                d = data[i + params["M"] - (keep_every_nth_frame - 1) * step:i + params["M"] + step]
                ax[0].plot(d[:,0], d[:,1], 'k.')

                ax[1].cla()
                ims = ax[1].imshow(bw.A, aspect='equal', interpolation='nearest')
                # fig.colorbar(ims)

                ax[0].set_title("Observation Model")
                ax[1].set_title("Transition Matrix")
                ax[1].set_xlabel("To")
                ax[1].set_ylabel("From")

                ax[1].set_xticks(np.arange(bw.N))
                live_nodes = [x for x in np.arange(bw.N) if x not in bw.dead_nodes]
                ax[1].set_yticks(live_nodes)

                moviewriter.grab_frame()
    if keep_every_nth_frame is not None:
        moviewriter.finish()
    return bw, moviewriter


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

def run_defaults(file):
    bw,_ = run_bubblewrap(file, default_parameters)
    plot_bubblewrap_results(bw)
    br,_ = BubblewrapRun(bw, file=file, bw_parameters=default_parameters)
    br.save()


if __name__ == "__main__":
    parameters = dict(
        num=8,
        lam=1e-3,
        nu=1e-3,
        eps=1e-4,
        step=8e-2,
        M=100,
        B_thresh=-5,
        batch=False,
        batch_size=1,
        go_fast=False,
        lookahead_steps=[1,2,4,8,16,32,64,128,256,512,1024],
        seed=42,
    )

    # file = "./generated/clock-04-18-18-12.npz"
    # file = "./generated/clock_switching_01.npz"
    # file = "./generated/clock_variable_01.npz"
    # file = "./generated/clock_wandering_01.npz"
    # file = "./generated/clock_steady_separated.npz"
    file = "./generated/clock-steadier_farther.npz"
    # file = "./generated/clock-slow_steadier_farther.npz"
    # file = "./generated/clock-halfspeed_farther.npz"
    # file = "./generated/clock-shuffled.npz"
    # file = "./generated/jpca_reduced.npy"
    # file = "./generated/neuropixel_reduced.npz"
    # file = "./generated/reduced_mouse.npy"
    # file = "./generated/widefield_reduced.npy"

    bw, _ = run_bubblewrap(file, parameters, keep_every_nth_frame=None, do_it_old_way=True)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters)
    br.save()

    bw, moviewriter = run_bubblewrap(file, parameters, keep_every_nth_frame=None, do_it_old_way=False)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters)
    br.save()

    if moviewriter is not None:
        old_fname = moviewriter.outfile.split(".")
        new_fname = br.outfile.split(".")

        new_fname[-1] = old_fname[-1]
        os.rename(moviewriter.outfile, ".".join(new_fname))

    plot_bubblewrap_results(bw)
