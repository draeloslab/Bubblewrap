# todo: maybe put JAX here to prevent gpu usage?
from bubblewrap import Bubblewrap
import datetime
import pickle
import os
from tqdm import tqdm
import numpy as np



class BWRun:
    def __init__(self, bw: Bubblewrap, data_source, animation_parameters=None, save_A=False):
        # todo: add total runtime tracker
        self.data_source = data_source
        self.bw = bw

        self.save_location = None
        self.total_runtime = None
        self.save_A = save_A
        self.animation_parameters = animation_parameters

        #todo: history_of object?
        self.prediction_history = {k : [] for k in data_source.time_offsets}
        self.entropy_history = {k : [] for k in data_source.time_offsets}
        self.behavior_pred_history = {k : [] for k in data_source.time_offsets}

        self.alpha_history = []
        self.n_living_history = []
        if save_A:
            self.A_history = []

        self.moviewriter = None

        assert (bw.d, bw.beh_dim) == data_source.get_pair_shapes()
        # note that if there is no behavior, the behavior dimensions will be zero

    def run(self):
        if self.animation_parameters is not None:
            self.start_animation()

        # todo: check initial dimensions
        for step, (obs, beh, offset_pairs) in enumerate(tqdm(self.data_source)):

            self.bw.observe(obs, beh)

            if step < self.bw.M:
                pass
            elif step == self.bw.M:
                self.bw.init_nodes()
                if self.animation_parameters is not None:
                    self.start_animation()
            else:
                self.log_for_step(step, offset_pairs)
                self.bw.e_step()
                self.bw.grad_Q()

        if self.animation_parameters is not None:
            self.finish_animation()

    def log_for_step(self, step, offset_pairs):
        # TODO: allow skipping of (e.g. entropy) steps?
        for offset, (o, b) in offset_pairs.items():
            p = self.bw.pred_ahead(self.bw.logB_jax(o, self.bw.mu, self.bw.L, self.bw.L_diag), self.bw.A, self.bw.alpha, offset)
            self.prediction_history[offset].append(p)

            e = self.bw.get_entropy(self.bw.A, self.bw.alpha, offset)
            self.entropy_history[offset].append(e)

            if self.bw.beh_dim:
                alpha_ahead = np.array(self.bw.alpha @ np.linalg.matrix_power(self.bw.A, offset)).reshape(-1,1)
                bp = self.bw.regressor.predict(alpha_ahead)

                self.behavior_pred_history[offset].append(bp)


        self.alpha_history.append(self.bw.alpha)
        self.n_living_history.append(self.bw.N - len(self.bw.dead_nodes))
        if self.save_A:
            self.A_history.append(self.bw.A)



        if self.animation_parameters is not None:
            self.moviewriter.grab_frame()

    def start_animation(self):
        pass
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10), layout='tight')
        # fig, ax = plt.subplots(1, 2, figsize=(10,7), layout='tight')

        # self.moviewriter = FFMpegFileWriter(fps=fps)
        # self.moviewriter.setup(fig, "generated/movie.mp4", dpi=100)

    def finish_animation(self):
        pass

    def save(self, directory="generated/bubblewrap_runs"):
        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.outfile = os.path.join(directory, f"bubblewrap_run_{time_string}.pickle")
        with open(self.outfile, "wb") as fhan:
            pickle.dump(self, fhan)
