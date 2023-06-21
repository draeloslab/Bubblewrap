import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib.animation import FFMpegFileWriter

from bubblewrap import Bubblewrap
from plot_2d_3d import plot_2d, plot_A_differences, plot_current_2d
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
    lookahead_steps=[1, 2, 5, 10],
    seed=42,
    save_A=False,
    balance=1,
    beh_reg_constant_term=True
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
    lookahead_steps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50],
    seed=42,
    save_A=False,
    balance=1,
    beh_reg_constant_term=True
)

def show_bubbles(ax, data, bw, params, step, i, keep_every_nth_frame):
    ax.cla()

    d = data[0:i+params["M"]+step]
    plot_2d(ax, d, bw)

    d = data[i + params["M"] - (keep_every_nth_frame - 1) * step:i + params["M"] + step]
    ax.plot(d[:,0], d[:,1], 'k.')
    ax.set_title(f"Observation Model (Bubbles) (i={i})")
    ax.set_xlabel("neuron 1")
    ax.set_ylabel("neuron 2")


def show_inhabited_bubbles(ax, data, bw, params, step, i, keep_every_nth_frame):
    ax.cla()

    d = data[0:i+params["M"]+step]
    plot_current_2d(ax, d, bw)

    d = data[i + params["M"] - (keep_every_nth_frame - 1) * step:i + params["M"] + step]
    ax.set_title(f"Currrent Bubbles (i={i})")
    ax.set_xlabel("neuron 1")
    ax.set_ylabel("neuron 2")

def show_A(ax, bw):
    ax.cla()
    ims = ax.imshow(bw.A, aspect='equal', interpolation='nearest')

    ax.set_title("Transition Matrix (A)")
    ax.set_xlabel("To")
    ax.set_ylabel("From")

    ax.set_xticks(np.arange(bw.N))
    live_nodes = [x for x in np.arange(bw.N) if x not in bw.dead_nodes]
    ax.set_yticks(live_nodes)

def show_D(ax, bw):
    ax.cla()
    ims = ax.imshow(bw.D, aspect='equal', interpolation='nearest')
    ax.set_title("D")

def show_Ct_y(ax, bw):
    old_ylim = ax.get_ylim()
    ax.cla()
    ax.plot(bw.Ct_y, '.-')
    ax.set_title("Ct_y")

    new_ylim = ax.get_ylim()
    ax.set_ylim([min(old_ylim[0], new_ylim[0]), max(old_ylim[1], new_ylim[1])])


def show_alpha(ax, bw):
    ax.cla()
    ims = ax.imshow(np.array(bw.alpha_list[-19:] + [bw.alpha]).T, aspect='auto', interpolation='nearest')

    ax.set_title("State Estimate ($\\alpha$)")
    live_nodes = [x for x in np.arange(bw.N) if x not in bw.dead_nodes]
    ax.set_yticks(live_nodes)
    ax.set_ylabel("bubble")
    ax.set_xlabel("steps (ago)")
    # ax.set_xticks([0.5,5,10,15,20])
    # ax.set_xticklabels([-20, -15, -10, -5, 0])
def show_behavior_variables(ax, bw, obs):
    ax.cla()
    ax.plot(bw.beh_list[-20:])
    ax.plot(obs[-20:])
    ax.set_ylim([-21,21])
    ax.set_title("Behavior prediction")

def show_A_eigenspectrum(ax, bw):
    ax.cla()
    eig = np.sort(np.linalg.eigvals(bw.A))[::-1]
    ax.plot(eig, '.')
    ax.set_title("Eigenspectrum of A")
    ax.set_ylim([0,1])

# TODO: remove this?
def mean_distance(data, shift=1):
    x = data - data.mean(axis=0)
    T = x.shape[0]

    differences = x[0:T - shift] - x[shift:T]
    distances = np.linalg.norm(differences, axis=1)

    return distances.mean()

def show_data_distance(ax, data, end_of_block, max_step=50):
    old_ylim = ax.get_ylim()
    ax.cla()
    start = max(end_of_block-3*max_step, 0)
    d = data[start:end_of_block]
    if d.shape[0] > 10:
        shifts = np.arange(0,min(d.shape[0]//2, max_step))
        distances = [mean_distance(d, shift) for shift in shifts]
        ax.plot(shifts, distances)
    ax.set_xlim([0,max_step])
    new_ylim = ax.get_ylim()
    ax.set_ylim([0, max(old_ylim[1], new_ylim[1])])
    ax.set_title(f"dataset[{start}:{end_of_block}] distances")
    ax.set_xlabel("offset")
    ax.set_ylabel("distance")

def show_nstep_pred_pdf(ax, bw, data, current_index, other_axis, fig, n=0):
    # vmin = np.inf
    # vmax = -np.inf
    if ax.collections:
        vmax = ax.collections[-3].colorbar.vmax
        vmin = ax.collections[-3].colorbar.vmin
        ax.collections[-3].colorbar.remove()
    ax.cla()
    other_axis: plt.Axes

    xlim = other_axis.get_xlim()
    ylim = other_axis.get_ylim()
    density = 50
    x_bins = np.linspace(*xlim, density+1)
    y_bins = np.linspace(*ylim, density+1)
    pdf = np.zeros(shape=(density, density))
    for i in range(density):
        for j in range(density):
            x = np.array([x_bins[i] + x_bins[i+1], y_bins[j] + y_bins[j+1]])/2
            b_values = bw.logB_jax(x, bw.mu, bw.L, bw.L_diag)
            pdf[i, j] = bw.alpha @ np.linalg.matrix_power(bw.A,n) @ np.exp(b_values)
    # cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T, vmin=min(vmin, pdf.min()), vmax=max(vmax, pdf.max()))
    # cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T, vmin=0, vmax=0.03) #log, vmin=-15, vmax=-5
    cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T) #log, vmin=-15, vmax=-5
    fig.colorbar(cmesh)
    if current_index+n < data.shape[0]:
            to_draw = data[current_index+n]
            ax.scatter(to_draw[0], to_draw[1], c='red', alpha=.25)

    to_draw = data[current_index]
    ax.scatter(to_draw[0], to_draw[1], c='red')
    ax.set_title(f"{n}-step pred. at t={current_index}")

def show_w(ax,bw):
    ax.cla()
    ax.plot(bw.D@bw.Ct_y, '.-')
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel("bubble #")
    ax.set_ylabel("weight magnitude")

def show_w_sideways(ax,bw, obs):
    ax.cla()
    w = np.array(bw.D @ bw.Ct_y)
    w[bw.dead_nodes] = 0

    a = np.array(bw.alpha)
    a = a / np.max(a)
    ax.plot(w, np.arange(w.size), alpha=0.25)
    ax.scatter(w, np.arange(w.size), alpha=a, c="C0")
    ylim = ax.get_ylim()
    ax.vlines(obs[-1], alpha=.5, ymin=ylim[0], ymax=ylim[1], colors="C1" )
    ax.set_ylabel("bubble #")
    ax.set_xlabel("weight magnitude")
    ax.set_title(r"Weights (times $\alpha$)")
    ax.set_xlim([-21, 21])

def run_bubblewrap(file, params, keep_every_nth_frame=None, do_it_old_way=False, end=None, tiles=1, movie_range=None,  invert_alternate_behavior=False, fps=20, behavior_shift=1, data_transform="n"):
    """this runs bubblewrap; it also generates a movie if `keep_every_nth_frame` is not None"""


    if keep_every_nth_frame is not None:
        fig, ax = plt.subplots(2, 2, figsize=(10,10), layout='tight')
        # fig, ax = plt.subplots(1, 2, figsize=(10,7), layout='tight')

        moviewriter = FFMpegFileWriter(fps=fps)
        moviewriter.setup(fig, "generated/movie.mp4", dpi=100)
    else:
        moviewriter = None

    s = np.load(file)

    if "npy" in file:
        data = s.T
    elif "npz" in file:
        data = s['y'][0]
        pre_obs = s['x']

        data = np.tile(data, reps=(tiles,1))
        obs = pre_obs
        for i in range(1,tiles):
            c = (-1)**i if invert_alternate_behavior else 1
            obs = np.hstack((obs, pre_obs*c))
    else:
        raise Exception("Unknown file extension.")


    obs = obs.reshape((-1,1))
    old_data = np.array(data)
    old_obs = np.array(obs)

    if data_transform == "n,b":
        data = np.hstack([data, obs])
    elif data_transform == "n":
        data = np.hstack([data])
    elif data_transform == "b":
        data = np.hstack([obs])
    else:
        raise Exception("You need to set data_transform")


    obs = np.hstack([old_data,old_obs])


    T = data.shape[0]       # should be big (like 20k)
    d = data.shape[1]       # should be small-ish (like 6)

    start = time.time()
    #todo:fix this
    params["behavior_shift"] = behavior_shift
    bw = Bubblewrap(d, beh_dim=obs.shape[1], **params)


    ## Set up for online run through dataset

    init = -params["M"]
    if end is None:
        # end = T-(params["M"] + max(bw.lookahead_steps))
        end = T
    else:
        if end < 0:
            end = T + end

    if end > T - behavior_shift:
        end = T - behavior_shift

    if movie_range is not None:
        movie_range = [m if m >=0 else T+m for m in movie_range]
        movie_range = np.array(movie_range) - params["M"]

    end = end - params["M"]


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
        bw.observe(data[start_of_block:end_of_block], obs[start_of_block+behavior_shift:end_of_block+behavior_shift])
        # if i > 800:
        # else:
        #     bw.observe(data[start_of_block:end_of_block], np.nan * obs[start_of_block+behavior_shift:end_of_block+behavior_shift])
        last_seen = end_of_block - 1

        future_observations = {}
        for x in bw.lookahead_steps:
            if (end_of_block - 1) + (x - 1) < T:
                future_observations[x] = data[(end_of_block - 1) + (x - 1)]

        bw.e_step(future_observations)
        bw.grad_Q()

        if keep_every_nth_frame is not None:
            assert step == 1 # the keep_every_th assumes the step is 1
            if i % keep_every_nth_frame == 0:
                if movie_range is None or (movie_range[0] <= i < movie_range[1]):
                    show_bubbles(ax[0,0], data, bw, params, step, i, keep_every_nth_frame)
                    show_behavior_variables(ax[1,1],bw,obs)
                    # show_w_sideways(ax[1,1], bw, obs[:end_of_block])
                    show_alpha(ax[1,0], bw)
                    # show_behavior_variables(ax[0,1], bw, obs[:end_of_block])
                    # show_w_sideways(ax[0,1],bw, obs)

                    # show_nstep_pred_pdf(ax[0,1], bw, data, last_seen, ax[0,0], fig, n=1)

                    # show_A_eigenspectrum(ax[1], bw)


                    moviewriter.grab_frame()
    end = time.time()
    if keep_every_nth_frame is not None:
        moviewriter.finish()

    return bw, moviewriter, end-start


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
    bw, _, _ = run_bubblewrap(file, parameters, keep_every_nth_frame=None, do_it_old_way=True, end=end)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters)
    br.save()
    old_file = br.outfile
    del br

    bw, moviewriter, _ = run_bubblewrap(file, parameters, keep_every_nth_frame=None, do_it_old_way=False, end=end)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters)
    br.save()
    new_file = br.outfile
    del br

    if shuffle is not False:
        shuffle_file = file.split("/")
        shuffle_file[-1] ="shuffled_" + shuffle_file[-1]
        shuffle_file = "/".join(shuffle_file)
        bw, moviewriter, time_spent = run_bubblewrap(shuffle_file, parameters, keep_every_nth_frame=None, do_it_old_way=False, end=end)
        br = BubblewrapRun(bw, file=shuffle_file, bw_parameters=parameters)
        br.save()
        s_file = br.outfile
        del br

        print(f"shuffled_new_way_file = '{s_file.split('/')[-1]}'")

    print(f"old_way_file = '{old_file.split('/')[-1]}'")
    print(f"new_way_file = '{new_file.split('/')[-1]}'")
    print(f"dataset = '{file.split('/')[-1]}'")


def simple_run(file, parameters, **kwargs):
    bw, moviewriter, time_spent = run_bubblewrap(file, parameters, **kwargs)
    # plot_bubblewrap_results(bw)
    br = BubblewrapRun(bw, file=file, bw_parameters=parameters, time_to_run=time_spent)
    br.save()

    if moviewriter is not None:
        old_fname = moviewriter.outfile.split(".")
        new_fname = br.outfile.split(".")

        new_fname[-1] = old_fname[-1]
        os.rename(moviewriter.outfile, ".".join(new_fname))
    print(br.outfile.split("/")[-1])


if __name__ == "__main__":
    file = "./generated/datasets/clock-steadier_farther.npz"

    # fewer nodes 1k, 100
    # bigger M
    # start at t=1000
    # run for longer (5k t)
    #

    simple_run(file,
               dict(default_rwd_parameters, num=50, M=100, step=8e-3, eps=1e-3, balance=1),
               keep_every_nth_frame=500, movie_range=None, fps=10,
               end=None, tiles=3, invert_alternate_behavior=True, behavior_shift=1, data_transform="n")