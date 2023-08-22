import numpy as np

from bubblewrap import Bubblewrap
from bubblewrap.default_parameters import default_clock_parameters
from bubblewrap.input_sources import NumpyDataSource
from bubblewrap.bw_run import BWRun, AnimationManager
import bubblewrap.plotting_functions as bpf

def test_can_run():
    rng = np.random.default_rng()
    m, n_obs, n_beh = 500, 3, 4
    obs = rng.normal(size=(m, n_obs))
    beh = rng.normal(size=(m, n_beh))
    ds = NumpyDataSource(obs, beh, time_offsets=(-10, 0, 10))

    bw = Bubblewrap(n_obs, n_beh, **default_clock_parameters)
    br = BWRun(bw, ds, show_tqdm=False)
    br.run()

def test_can_run_without_beh():
    rng = np.random.default_rng()
    m, n_obs, n_beh = 500, 3, 4
    obs = rng.normal(size=(m, n_obs))
    ds = NumpyDataSource(obs, time_offsets=(-10, 0, 10))

    bw = Bubblewrap(3, **default_clock_parameters)
    br = BWRun(bw, ds, show_tqdm=False)
    br.run()

def test_can_make_video():
    rng = np.random.default_rng()
    m, n_obs, n_beh = 500, 3, 4
    obs = rng.normal(size=(m, n_obs))
    ds = NumpyDataSource(obs, time_offsets=(-10, 0, 10))

    class CustomAnimation(AnimationManager):
        n_rows = 1
        n_cols = 1
        def custom_draw_frame(self, step, bw, br):
            bpf.show_A(self.ax, bw) # note: you would usually index into ax, but this call uses 1 row and 1 column

    ca = CustomAnimation()

    bw = Bubblewrap(3, **default_clock_parameters)
    br = BWRun(bw, ds, ca, show_tqdm=False)
    br.run()
