import time
import numpy as np

import matplotlib
import matplotlib.pylab as plt
from matplotlib.animation import FFMpegFileWriter

from bubblewrap import Bubblewrap, BWRun
from bubblewrap.default_parameters import default_clock_parameters, default_rwd_parameters
from bubblewrap.plotting_functions import plot_2d, plot_A_differences, plot_current_2d, show_bubbles, show_behavior_variables, show_alpha

from math import floor
from tqdm import tqdm

import os
if os.environ.get("display") is not None:
    matplotlib.use('QtAgg')



def run_bubblewrap(data, params, keep_every_nth_frame=None, do_it_old_way=False, end=None, tiles=1, movie_range=None,  invert_alternate_behavior=False, fps=20, behavior_shift=1, data_transform="n"):
    """this runs bubblewrap; it also generates a movie if `keep_every_nth_frame` is not None"""

    T = data.shape[0] # should be big (like 20k)
    d = data.shape[1] # should be small-ish (like 6)

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

    # Run online, 1 data or batch at a time
    for i in tqdm(np.arange(init, end, step)):
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
    br = BWRun(bw, file=file, bw_parameters=parameters)
    br.save()
    old_file = br.outfile
    del br

    bw, moviewriter, _ = run_bubblewrap(file, parameters, keep_every_nth_frame=None, do_it_old_way=False, end=end)
    br = BWRun(bw, file=file, bw_parameters=parameters)
    br.save()
    new_file = br.outfile
    del br

    if shuffle is not False:
        shuffle_file = file.split("/")
        shuffle_file[-1] ="shuffled_" + shuffle_file[-1]
        shuffle_file = "/".join(shuffle_file)
        bw, moviewriter, time_spent = run_bubblewrap(shuffle_file, parameters, keep_every_nth_frame=None, do_it_old_way=False, end=end)
        br = BWRun(bw, file=shuffle_file, bw_parameters=parameters)
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
    br = BWRun(bw, file=file, bw_parameters=parameters, time_to_run=time_spent)
    br.save()

    if moviewriter is not None:
        old_fname = moviewriter.outfile.split(".")
        new_fname = br.outfile.split(".")

        new_fname[-1] = old_fname[-1]
        os.rename(moviewriter.outfile, ".".join(new_fname))
    print(br.outfile.split("/")[-1])


if __name__ == "__main__":
    file = "./generated/datasets/clock-halfspeed_farther.npz"