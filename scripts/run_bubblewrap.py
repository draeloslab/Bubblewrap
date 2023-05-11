import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib.animation import FFMpegFileWriter

from bubblewrap import Bubblewrap
from plot_2d_3d import plot_2d, plot_A_differences
from bubblewrap_run import BubblewrapRun

from math import atan2, floor
from tqdm import tqdm

matplotlib.use('QtAgg')


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

default_rwd_parameters = dict(
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
    lookahead_steps=[1, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128],
    seed=42,
    save_A=False,
)

default_clock_parameters = dict(
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
    lookahead_steps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 32, 35, 40, 50, 64, 80, 100, 128, 256, 512],
    seed=42,
    save_A=False,
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

    if "npy" in file:
        data = s.T
    elif "npz" in file:
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



def compare_old_and_new_ways(file, parameters, end=None, shuffle=True):
    bw, _ = run_bubblewrap(file, parameters, keep_every_nth_frame=None, do_it_old_way=True, end=end)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters)
    br.save()
    old_file = br.outfile
    del br

    bw, moviewriter = run_bubblewrap(file, parameters, keep_every_nth_frame=None, do_it_old_way=False, end=end)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters)
    br.save()
    new_file = br.outfile
    del br

    if shuffle is not False:
        shuffle_file = file.split("/")
        shuffle_file[-1] ="shuffled_" + shuffle_file[-1]
        shuffle_file = "/".join(shuffle_file)
        bw, moviewriter = run_bubblewrap(shuffle_file, parameters, keep_every_nth_frame=None, do_it_old_way=False, end=end)
        br = BubblewrapRun(bw, file=shuffle_file, bw_parameters=parameters)
        br.save()
        s_file = br.outfile
        del br

        print(f"shuffled_new_way_file = '{s_file.split('/')[-1]}'")

    print(f"old_way_file = '{old_file.split('/')[-1]}'")
    print(f"new_way_file = '{new_file.split('/')[-1]}'")
    print(f"dataset = '{file.split('/')[-1]}'")

def simple_run(file, parameters, nth_frame=10):
    bw, moviewriter = run_bubblewrap(file, parameters, keep_every_nth_frame=nth_frame)
    # plot_bubblewrap_results(bw)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters)
    br.save()

    if moviewriter is not None:
        old_fname = moviewriter.outfile.split(".")
        new_fname = br.outfile.split(".")

        new_fname[-1] = old_fname[-1]
        os.rename(moviewriter.outfile, ".".join(new_fname))
    print(br.outfile.split("/")[-1])


def compare_new_and_old_way_main():
    compare_old_and_new_ways("./generated/clock-steadier_farther.npz", dict(default_clock_parameters, save_A=True))
    compare_old_and_new_ways("./generated/clock-slow_steadier_farther.npz", dict(default_clock_parameters, save_A=True))
    compare_old_and_new_ways("./generated/clock-halfspeed_farther.npz", dict(default_clock_parameters, save_A=True))
    compare_old_and_new_ways("./generated/clock-shuffled.npz", dict(default_clock_parameters, save_A=True))

    compare_old_and_new_ways("./generated/jpca_reduced.npy", dict(default_rwd_parameters, save_A=True))
    compare_old_and_new_ways("./generated/neuropixel_reduced.npz", dict(default_rwd_parameters, save_A=True), end=10_00)
    compare_old_and_new_ways("./generated/reduced_mouse.npy", dict(default_rwd_parameters, save_A=True), end=10_000)
    compare_old_and_new_ways("./generated/widefield_reduced.npy", dict(default_rwd_parameters, save_A=True), end=10_000)


    compare_old_and_new_ways("./generated/neuropixel_reduced.npz", dict(default_rwd_parameters))
    compare_old_and_new_ways("./generated/widefield_reduced.npy", dict(default_rwd_parameters))

    compare_old_and_new_ways("./generated/reduced_mouse.npy", dict(default_rwd_parameters))

if __name__ == "__main__":
    file = "./generated/datasets/jpca_reduced.npy"
    # `num` is set by the entropy plot
    # `step` \in [10e-3, and 10e-1]
    # `eps` \in {0, 1e-3}
    # simple_run(file, parameters=dict(default_rwd_parameters, num=100, M=21*2, eps=1e-3*1, step=.2), nth_frame=None)

    for step in [.001, .01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .8,]:
    # for step in [.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, ]:
        simple_run(file, parameters=dict(default_rwd_parameters, num=100, M=21*2, eps=0,    step=step, B_thresh=-15, nu=1e-3, lam=1e-3,), nth_frame=None)
        simple_run(file, parameters=dict(default_rwd_parameters, num=100, M=21*2, eps=1e-3, step=step, B_thresh=-15, nu=1e-3, lam=1e-3,), nth_frame=None)

# eps 1e-3
# step 9e-2, 9e-3

# lam = 1e-3          # lambda
# nu = 1e-3           # nu
# eps = 1e-3          # epsilon sets data forgetting
# step = 8e-2         # for adam gradients
# B_thresh = -10      # threshold for when to teleport (log scale)

# n = 1000             # number of nodes to tile with
# lam = 1e-3          # lambda
# nu = 1e-3           # nu
# eps = 1e-3          # epsilon sets data forgetting
# step = 8e-2         # for adam gradients
# m = 30              # small set of data seen for initialization
# b_thresh = -10      # threshold for when to teleport (log scale)