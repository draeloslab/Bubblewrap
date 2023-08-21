import numpy as np
import pickle
from bubblewrap.input_sources import NumpyDataSource
from bubblewrap import Bubblewrap
from bubblewrap import BWRun
from bubblewrap.default_parameters import default_clock_parameters

if __name__ == '__main__':

    file = "./generated/datasets/clock-halfspeed_farther.npz"
    f = np.load(file)
    obs = np.squeeze(f["y"])
    beh = f["x"].reshape([-1,1])

    ds = NumpyDataSource(obs, beh, time_offsets=(1,))
    obs_dim, beh_dim = ds.get_pair_shapes()

    bw = Bubblewrap(obs_dim, beh_dim, **default_clock_parameters)
    br = BWRun(bw, ds)
    br._run_all_data()

    br.save()
