# todo: maybe put JAX here to prevent gpu usage?
from bubblewrap import Bubblewrap
import datetime
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
import warnings

class AnimationManager:
    # todo: this could inherit from FileWriter; that might be better design
    def __init__(self, n_rows=2, n_cols=2, fps=20, dpi=100, outfile="generated/movie.mp4"):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fps = fps
        self.dpi = dpi
        self.outfile = outfile

        self.movie_writer = FFMpegFileWriter(fps=self.fps)

        self.fig, self.ax = plt.subplots(self.n_rows, self.n_cols, figsize=(10, 10), layout='tight')

        self.movie_writer.setup(self.fig, self.outfile, dpi=self.dpi)

        self.finished = False

        self.final_output_location = None

    def set_final_output_location(self, final_output_location):
        self.final_output_location = final_output_location

    def finish(self):
        if not self.finished:
            self.movie_writer.finish()
            os.rename(self.outfile, self.final_output_location)
            self.finished = True

    def frame_draw_condition(self, frame_number, bw):
        return True

    def draw_frame(self, step, bw, br):
        self.custom_draw_frame(step, bw, br)
        self.movie_writer.grab_frame()

    def custom_draw_frame(self, step, bw, br):
        pass

class BWRun:
    def __init__(self, bw: Bubblewrap, data_source, animation_manager=None, save_A=False, show_tqdm=True, output_directory="generated/bubblewrap_runs"):
        # todo: add total runtime tracker
        self.data_source = data_source
        self.bw = bw
        self.animation_manager : AnimationManager = animation_manager

        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.output_prefix = os.path.join(output_directory, f"bubblewrap_run_{time_string}")
        self.pickle_file = f"{self.output_prefix}.pickle"
        if self.animation_manager:
            self.animation_manager.set_final_output_location(f"{self.output_prefix}.mp4")

        # self.total_runtime = None
        self.save_A = save_A
        self.show_tqdm = show_tqdm

        #todo: history_of object?
        self.prediction_history = {k: [] for k in data_source.time_offsets}
        self.entropy_history = {k: [] for k in data_source.time_offsets}
        self.behavior_pred_history = {k: [] for k in data_source.time_offsets}

        self.alpha_history = []
        self.n_living_history = []
        if save_A:
            self.A_history = []

        self.saved = False

        assert (bw.d, bw.beh_dim) == data_source.get_pair_shapes()
        # note that if there is no behavior, the behavior dimensions will be zero

    def run(self, save=True):
        if len(self.data_source) < self.bw.M:
            warnings.warn("Data length shorter than initialization.")

        f = tqdm if self.show_tqdm else lambda x: x
        for step, (obs, beh, offset_pairs) in enumerate(f(self.data_source)):
            self.bw.observe(obs, beh)

            if step < self.bw.M:
                pass
            elif step == self.bw.M:
                self.bw.init_nodes()
                self.bw.e_step() # todo: is this OK?
                self.bw.grad_Q()
            else:
                self.log_for_step(step, offset_pairs)
                self.bw.e_step()
                self.bw.grad_Q()

        if self.animation_manager:
            self.animation_manager.finish()
        if save:
            self.save()


    def _run_all_data(self):

        for i in range(self.bw.M):
            obs, beh = self.data_source[i]
            self.bw.observe(obs, beh)

        self.bw.init_nodes()

        for i in range(len(self.data_source)):
            obs, beh = self.data_source[i]

            offset_pairs = {}
            for offset in self.data_source.time_offsets:
                offset_pairs[offset] = self.data_source[i + offset]

            self.bw.observe(obs, beh)

            self.log_for_step(i, offset_pairs)
            self.bw.e_step()
            self.bw.grad_Q()



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

        if self.animation_manager and self.animation_manager.frame_draw_condition(step, self.bw):
            self.animation_manager.draw_frame(step, self.bw, self)


    def finish_and_remove_jax(self):
        if self.animation_manager:
            self.animation_manager.finish()
            del self.animation_manager

        def convert_dict(d):
            return {k:np.array(v) for k, v in d.items()}

        self.prediction_history = convert_dict(self.prediction_history)
        self.entropy_history = convert_dict(self.entropy_history)
        self.behavior_pred_history = convert_dict(self.behavior_pred_history)

        self.alpha_history = np.array(self.alpha_history)
        self.n_living_history = np.array(self.n_living_history)
        if self.save_A:
            self.A_history = np.array(self.A_history)

        self.bw.freeze()


    def save(self,  freeze=True):
        self.saved = True
        if freeze:
            self.finish_and_remove_jax()

        with open(self.pickle_file, "wb") as fhan:
            # todo: remove this, it's just to make the comparison fair
            del self.data_source
            pickle.dump(self, fhan)

    def __del__(self):
        if not self.saved:
            warnings.warn("A Bubblewrap run was not saved.")
