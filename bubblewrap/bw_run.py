# todo: maybe put JAX here to prevent gpu usage?
from bubblewrap import Bubblewrap
from input_sources import NumpyDataSource
import time
import datetime
import pickle
import os
from tqdm import tqdm



class BWRun:
    def __init__(self, bw: Bubblewrap, data_source, animation_parameters=None):
        # todo: add total runtime tracker
        self.data_source = data_source
        self.bw = bw

        self.save_location = None
        self.total_runtime = None
        self.animation_parameters = animation_parameters

        self.prediction_history = {k : [] for k in data_source.time_offsets}
        self.entropy_history = {k : [] for k in data_source.time_offsets}
        self.moviewriter = None

        assert bw.d, bw.b_d == data_source.get_pair_shapes()

    def run(self):
        if self.animation_parameters is not None:
            self.start_animation()

        # todo: check initial dimensions
        for step, (datapoint, behaviorpoint, offset_pairs) in enumerate(tqdm(self.data_source)):

            bw.observe(datapoint)

            if step < self.bw.M:
                pass
            elif step == self.bw.M:
                self.bw.init_nodes()
                if self.animation_parameters is not None:
                    self.start_animation()
            else:
                self.log_for_step(step, offset_pairs)
                self.bw.e_step(future_observations={1:offset_pairs[1][0]})
                self.bw.grad_Q()

        if self.animation_parameters is not None:
            self.finish_animation()

    def start_animation(self):
        pass
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10), layout='tight')
        # fig, ax = plt.subplots(1, 2, figsize=(10,7), layout='tight')

        # self.moviewriter = FFMpegFileWriter(fps=fps)
        # self.moviewriter.setup(fig, "generated/movie.mp4", dpi=100)

    def finish_animation(self):
        pass


    def log_for_step(self, step, offset_pairs):
        for offset, (o, b) in offset_pairs.items():
            p = self.bw.pred_ahead(self.bw.logB_jax(o, self.bw.mu, self.bw.L, self.bw.L_diag), self.bw.A, self.bw.alpha, offset)
            self.prediction_history[offset].append(p)

            e = self.bw.get_entropy(self.bw.A, self.bw.alpha, offset)
            self.entropy_history[offset].append(e)

        if self.animation_parameters is not None:
            self.moviewriter.grab_frame()


    def save(self, directory="generated/bubblewrap_runs"):
        time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.outfile = os.path.join(directory, f"bubblewrap_run_{time_string}.pickle")
        with open(self.outfile, "wb") as fhan:
            pickle.dump(self, fhan)



if __name__ == '__main__':
    from default_parameters import default_clock_parameters
    import numpy as np
    bw = Bubblewrap(3,3, **dict(default_clock_parameters, lookahead_steps=[1]))

    rng = np.random.default_rng()
    m, n = 500, 3
    obs = rng.normal(size=(m,n))
    beh = rng.normal(size=(m,n))
    ds = NumpyDataSource(obs, beh, time_offsets=(1,))

    br = BWRun(bw, ds)
    br.run()
    print(np.array(br.bw.entropy_list)[:,0] - np.array(br.entropy_history[1]))
    print(np.array(br.bw.pred_list)[:, 0] - np.array(br.prediction_history[1]))
