from bubblewrap import Bubblewrap
import datetime
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegFileWriter
from .input_sources.functional import save_to_cache
from .input_sources.data_sources import NumpyPairedDataSource
import warnings
import time
from .config import CONFIG

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_sources.data_sources import NumpyPairedDataSource, ConsumableDataSource
    from .regressions import OnlineRegressor

@save_to_cache("simple_bw_run")
def simple_bw_run(input_array, bw_params, time_offsets=(1,)):
    ds = NumpyPairedDataSource(input_array, time_offsets=time_offsets)
    bw = Bubblewrap(input_array.shape[1], **bw_params)
    br = BWRun(bw, ds, show_tqdm=True)
    br.run(save=True)
    return br

class BWRun:
    def __init__(self, bw, data_source, behavior_regressors=(), animation_manager=None, save_A=False, show_tqdm=True,
                 output_directory=CONFIG["output_path"]/"bubblewrap_runs"):
        # todo: add total runtime tracker
        self.data_source: ConsumableDataSource = data_source
        self.bw: Bubblewrap = bw
        self.animation_manager: AnimationManager = animation_manager

        # only keep a behavior regressor if there is behavior
        self.behavior_regressors = []
        if self.data_source.output_shape[1] > 0:
            self.behavior_regressors: [OnlineRegressor] = behavior_regressors


        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.output_prefix = os.path.join(output_directory, f"bubblewrap_run_{time_string}")
        self.pickle_file = f"{self.output_prefix}.pickle"
        if self.animation_manager:
            self.animation_manager.set_final_output_location(f"{self.output_prefix}.{self.animation_manager.extension}")

        # self.total_runtime = None
        self.save_A = save_A
        self.show_tqdm = show_tqdm

        # todo: history_of object?
        self.prediction_history = {k: [] for k in data_source.time_offsets}
        self.entropy_history = {k: [] for k in data_source.time_offsets}
        self.behavior_pred_history = [{k: [] for k in data_source.time_offsets} for i in range(len(self.behavior_regressors))]
        self.behavior_error_history = [{k: [] for k in data_source.time_offsets} for i in range(len(self.behavior_regressors))]
        self.alpha_history = {k: [] for k in data_source.time_offsets}
        self.runtime = None

        self.n_living_history = []
        if save_A:
            self.A_history = []

        self.saved = False
        self.frozen = False

        obs_dim, beh_dim = self.data_source.output_shape
        assert obs_dim == self.bw.d
        if self.behavior_regressors:
            for r in self.behavior_regressors:
                assert beh_dim == r.output_d
        # note that if there is no behavior, the behavior dimensions will be zero

    def run(self, save=False, limit=None, freeze=True):
        start_time = time.time()

        if len(self.data_source) < self.bw.M:
            warnings.warn("Data length shorter than initialization.")

        if limit is None:
            limit = len(self.data_source)
        limit = min(len(self.data_source), limit)

        # todo: make a wrapper around data streams that works with TQDM
        generator = tqdm(self.data_source.triples(limit=limit),
                         total=limit) if self.show_tqdm else self.data_source.triples(limit=limit)
        for step, (obs, beh, offset_pairs) in enumerate(generator):
            self.bw.observe(obs)

            if step < self.bw.M:
                pass
            elif step == self.bw.M:
                self.bw.init_nodes()
                self.bw.e_step()  # todo: is this OK?
                self.bw.grad_Q()
            else:
                self.bw.e_step()
                self.bw.grad_Q()

                if self.behavior_regressors:
                    for i in range(len(self.behavior_regressors)):
                        self.behavior_regressors[i].safe_observe(self.bw.alpha, beh)
                self.log_for_step(step, offset_pairs)

        self.runtime = time.time() - start_time

        if freeze:
            self.finish_and_remove_jax()
        if save:
            self.saved = True
            with open(self.pickle_file, "wb") as fhan:
                pickle.dump(self, fhan)

    def log_for_step(self, step, offset_pairs):
        # TODO: allow skipping of (e.g. entropy) steps?
        for offset, (o, b) in offset_pairs.items():
            p = self.bw.pred_ahead(self.bw.logB_jax(o, self.bw.mu, self.bw.L, self.bw.L_diag), self.bw.A, self.bw.alpha,
                                   offset)
            self.prediction_history[offset].append(p)

            e = self.bw.get_entropy(self.bw.A, self.bw.alpha, offset)
            self.entropy_history[offset].append(e)
            self.alpha_history[offset].append(self.bw.alpha @ np.linalg.matrix_power(self.bw.A, offset))

            if self.behavior_regressors:
                for i in range(len(self.behavior_regressors)):
                    alpha_ahead = np.array(self.bw.alpha @ np.linalg.matrix_power(self.bw.A, offset)).reshape(-1, 1)
                    bp = self.behavior_regressors[i].predict(alpha_ahead)

                    self.behavior_pred_history[i][offset].append(bp)
                    self.behavior_error_history[i][offset].append(bp - b)

        self.n_living_history.append(self.bw.N - len(self.bw.dead_nodes))
        if self.save_A:
            self.A_history.append(self.bw.A)

        if self.animation_manager and self.animation_manager.frame_draw_condition(step, self.bw):
            self.animation_manager.draw_frame(step, self.bw, self)

    def finish_and_remove_jax(self):
        self.frozen = True
        if self.animation_manager:
            self.animation_manager.finish()
            del self.animation_manager

        def convert_dict(d):
            return {k: np.array(v) for k, v in d.items()}

        self.prediction_history = convert_dict(self.prediction_history)
        self.entropy_history = convert_dict(self.entropy_history)
        self.behavior_pred_history = [convert_dict(x) for x in self.behavior_pred_history]
        self.behavior_error_history = [convert_dict(x) for x in self.behavior_error_history]
        self.alpha_history = convert_dict(self.alpha_history)

        self.n_living_history = np.array(self.n_living_history)
        if self.save_A:
            self.A_history = np.array(self.A_history)

        self.bw.freeze()

    # Metrics
    def evaluate_regressor(self, reg, o, train_offset=0, test_offset=1):
        train = self.alpha_history[train_offset]
        test = self.alpha_history[test_offset]
        pred = []

        for i, (x, y) in enumerate(list(zip(train, o))[:-test_offset]):  # TODO: that `:-test_offset` is not thought out
            reg.safe_observe(x, y)
            pred.append(reg.predict(test[i]))

        pred = np.array(pred)
        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1)

        return pred, o[test_offset:] # predicted, true

    def _last_half_index(self):
        assert self.frozen
        assert self.data_source.time_offsets
        pred = self.prediction_history[self.data_source.time_offsets[0]]
        assert np.isfinite(pred)
        return len(pred)//2

    def behavior_pred_corr(self, offset):
        pred, true, err = self.get_behavior_last_half(offset)
        assert np.all(np.isfinite(true))
        return np.corrcoef(pred, true)[0,1]

    def get_behavior_last_half(self, offset):
        i = self._last_half_index()
        pred = self.behavior_pred_history[offset][-i:]
        err = self.behavior_error_history[offset][-i:]
        true = pred - err
        return pred, true, err

    def log_pred_p_summary(self, offset):
        i = self._last_half_index()
        return self.prediction_history[offset][-i:].mean()

    def entropy_summary(self, offset):
        i = self._last_half_index()
        return self.entropy_history[offset][-i:].mean()


class AnimationManager:
    # todo: this could inherit from FileWriter; that might be better design
    n_rows = 2
    n_cols = 2
    fps = 20
    dpi = 100
    extension = "mp4"
    outfile = f"./movie.{extension}"
    figsize = (10, 10)

    def __init__(self):
        self.movie_writer = FFMpegFileWriter(fps=self.fps)
        self.fig, self.ax = plt.subplots(self.n_rows, self.n_cols, figsize=self.figsize, layout='tight')
        self.movie_writer.setup(self.fig, self.outfile, dpi=self.dpi)
        self.finished = False
        self.final_output_location = None
        self.setup()

    def setup(self):
        pass

    def set_final_output_location(self, final_output_location):
        self.final_output_location = final_output_location

    def finish(self):
        if not self.finished:
            self.movie_writer.finish()
            os.rename(self.outfile, self.final_output_location)
            self.finished = True

    def frame_draw_condition(self, step_number, bw):
        return True

    def draw_frame(self, step, bw, br):
        self.custom_draw_frame(step, bw, br)
        self.movie_writer.grab_frame()

    def custom_draw_frame(self, step, bw, br):
        pass
