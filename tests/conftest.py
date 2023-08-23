import pytest
import numpy as np
from bubblewrap import Bubblewrap, NumpyDataSource, BWRun, SymmetricNoisy, default_clock_parameters

@pytest.fixture
def outdir(tmpdir):
    tmpdir.mkdir("generated")
    outdir = tmpdir.mkdir("generated/bubblewrap_runs")
    return outdir


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def make_br(rng, outdir):
    def br_f():
        m, n_obs, n_beh = 200, 2, 3
        obs = rng.normal(size=(m, n_obs))
        beh = rng.normal(size=(m, n_beh))
        ds = NumpyDataSource(obs, beh, time_offsets=(3, 0, 3))

        bw = Bubblewrap(n_obs, **default_clock_parameters)
        reg = SymmetricNoisy(bw.N, n_beh, forgetting_factor=1 - (1e-2), noise_scale=1e-5)
        br = BWRun(bw, ds, behavior_regressor=reg, show_tqdm=True, output_directory=outdir)
        br.run(save=True)
        return br
    return br_f
