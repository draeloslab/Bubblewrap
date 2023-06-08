from bubblewrap import Bubblewrap
import numpy as np
import datetime
import pickle
import os

class BubblewrapRun:
    def __init__(self, bw: Bubblewrap, file, bw_parameters=None, time_to_run=None):
        self.file = file
        self.bw_parameters = bw_parameters
        self.time_to_run = time_to_run
        self.outfile = None

        self.A = np.array(bw.A)
        self.mu = np.array(bw.mu)
        self.L = np.array(bw.L)
        self.n_obs = np.array(bw.n_obs)
        self.pred_list = np.array(bw.pred_list)
        self.beh_regret_list = np.array(bw.beh_regret_list)
        self.beh_list = np.array(bw.beh_list)
        self.beh_counts = np.array(bw.beh_counts)
        self.D = np.array(bw.D)
        self.Ct_y = np.array(bw.Ct_y)
        self.entropy_list = np.array(bw.entropy_list)
        if hasattr(bw, "alpha_list"):
            # todo: check if I need this guard
            self.alpha_list = np.array(bw.alpha_list)

        if hasattr(bw, "A_list"):
            self.A_list = np.array(bw.A_list)
        self.dead_nodes = np.array(bw.dead_nodes)

    def save(self, directory="generated/bubblewrap_runs"):
        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.outfile = os.path.join(directory, f"bubblewrap_run_{time_string}.pickle")
        with open(self.outfile, "wb") as fhan:
            pickle.dump(self, fhan)
