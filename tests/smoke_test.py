import numpy as np

from bubblewrap import Bubblewrap
from bubblewrap.default_parameters import default_clock_parameters
from bubblewrap.input_sources import NumpyDataSource
from bubblewrap.bw_run import BWRun

def test_can_run():
    rng = np.random.default_rng()
    m, n_obs, n_beh = 100, 3, 4
    obs = rng.normal(size=(m, n_obs))
    beh = rng.normal(size=(m, n_beh))
    ds = NumpyDataSource(obs, beh, time_offsets=(-10, 0, 10))

    bw = Bubblewrap(n_obs, n_beh, **default_clock_parameters)
    br = BWRun(bw, ds)
    br.run()

def test_can_run_without_beh():
    rng = np.random.default_rng()
    m, n_obs, n_beh = 100, 3, 4
    obs = rng.normal(size=(m, n_obs))
    ds = NumpyDataSource(obs, time_offsets=(-10, 0, 10))

    bw = Bubblewrap(3, **default_clock_parameters)
    br = BWRun(bw, ds)
    br.run()