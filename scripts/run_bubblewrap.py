import time
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import pickle
import os
import datetime
from bubblewrap import Bubblewrap
from math import floor
from tqdm import tqdm

matplotlib.use("QtAgg")


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
    num=100,
    lam=1e-3,
    nu=1e-3,
    eps=1e-3,
    step=8e-2,
    M=30,
    B_thresh=-10,
    batch=False,
    batch_size=1,
    go_fast=False,
    lookahead_steps=[1, 10],
    seed=42,
)

generated_files = [
    "./generated/lorenz_1trajectories_3dim_500to20500_noise0.2.npz",
    "./generated/lorenz_1trajectories_3dim_500to20500_noise0.05.npz",
    "./generated/vdp_1trajectories_2dim_500to20500_noise0.2.npz",
    "./generated/vdp_1trajectories_2dim_500to20500_noise0.05.npz",
]
default_file = generated_files[2]


def run_bubblewrap(file, params):
    s = np.load(file)
    data = s["y"][0]

    T = data.shape[0]  # should be 20k
    d = data.shape[1]  # should be 2

    bw = Bubblewrap(d, **params)

    # Set up for online run through dataset

    init = -params["M"]
    end = T - params["M"]
    step = params["batch_size"]

    # Initialize things
    for i in np.arange(0, params["M"], step):
        if params["batch"]:
            bw.observe(data[i : i + step])
        else:
            bw.observe(data[i])
    bw.init_nodes()

    # Run online, 1 data or batch at a time
    for i in tqdm(np.arange(init, end, step)):
        bw.observe(data[i + params["M"] : i + params["M"] + step])
        bw.e_step()
        bw.grad_Q()
    return bw


def plot_bubblewrap_results(bw, running_average_length=500):
    T = len(bw.pred_list)

    pred_mat = np.array(bw.pred_list)
    ent_mat = np.array(bw.entropy_list)

    for step_n in range(len(bw.lookahead_steps)):
        steps = bw.lookahead_steps[step_n]
        fig, ax = plt.subplots(1, 2, sharex="all")
        pred = pred_mat[:, step_n]
        ax[0].plot(pred)
        print(f"Mean pred ahead ({steps} steps): {np.mean(pred[-floor(T/2):])}")

        var_tmp = np.convolve(
            pred, np.ones(running_average_length) / running_average_length, mode="valid"
        )
        var_tmp_x = np.arange(var_tmp.size) + running_average_length // 2
        ax[0].plot(var_tmp_x, var_tmp, "k")
        ax[0].set_title(f"Prediction ({steps} steps)")

        ent = ent_mat[:, step_n]
        ax[1].plot(ent)
        print(f"Mean entropy ({steps} steps): {np.mean(ent[-floor(T/2):])}")
        ax[1].plot(var_tmp_x, var_tmp, "k")
        ax[1].set_title(f"Entropy ({steps} steps)")
    plt.show()


def generate_random_bw_hyperparameters(variable_parameters=None):
    rng = np.random.default_rng()

    if variable_parameters is None:
        variable_parameters = dict(
            num=[16, 256, 1024],
            lam=[1e-4, 1e-3, 1e-3, 1e-3, 1e-2],
            nu=[1e-4, 1e-3, 1e-3, 1e-3, 1e-2],
            eps=[1e-4, 1e-3, 1e-2],
            B_thresh=[-15, -10, -5],
            seed=[10 * x for x in range(100)],
        )
    while True:
        d = dict(default_parameters)
        for key, value in variable_parameters.items():
            d[key] = rng.choice(value)
        f = rng.choice(generated_files)
        yield d, f


class BubblewrapRun:
    def __init__(self, bw: Bubblewrap, file, bw_parameters=None, time_to_run=None):
        self.file = file
        self.bw_parameters = bw_parameters
        self.time_to_run = time_to_run

        self.A = np.array(bw.A)
        self.mu = np.array(bw.mu)
        self.L = np.array(bw.L)
        self.n_obs = np.array(bw.n_obs)
        # TODO: calling these lists is misleading
        self.pred_list = np.array(bw.pred_list)
        self.entropy_list = np.array(bw.entropy_list)
        self.dead_nodes = np.array(bw.dead_nodes)

    def save(self, directory="generated/bubblewrap_runs"):
        time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = os.path.join(directory, f"bubblewrap_run_{time_string}.pickle")
        with open(filename, "wb") as fhan:
            pickle.dump(self, fhan)


def run_defaults():
    bw = run_bubblewrap(default_file, default_parameters)
    plot_bubblewrap_results(bw)
    br = BubblewrapRun(bw, file=default_file, bw_parameters=default_parameters)
    br.save()


def do_many_random_runs():
    for p, f in generate_random_bw_hyperparameters():
        try:
            start_time = time.time()
            bw = run_bubblewrap(default_file, p)
            end_time = time.time()
            br = BubblewrapRun(
                bw, file=f, bw_parameters=p, time_to_run=end_time - start_time
            )
            br.save()
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e


if __name__ == "__main__":
    run_defaults()
