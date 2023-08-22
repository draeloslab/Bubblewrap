import matplotlib.pyplot as plt
import pytest
import numpy as np

from bubblewrap import Bubblewrap
from bubblewrap.default_parameters import default_clock_parameters
from bubblewrap.input_sources import NumpyDataSource
from bubblewrap.bw_run import BWRun
import bubblewrap.plotting_functions as bpf
from bubblewrap.regressions import SymmetricNoisy

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def make_br(rng):
    def br_f():
        m, n_obs, n_beh = 200, 2, 3
        obs = rng.normal(size=(m, n_obs))
        beh = rng.normal(size=(m, n_beh))
        ds = NumpyDataSource(obs, beh, time_offsets=(3, 0, 3))

        bw = Bubblewrap(n_obs, **default_clock_parameters)
        reg = SymmetricNoisy(bw.N, n_beh, forgetting_factor=1 - (1e-2), noise_scale=1e-5)
        br = BWRun(bw, ds, behavior_regressor=reg, show_tqdm=False)
        br.run(save=True)
        return br
    return br_f

def test_axis_plots(make_br):

    fig, axs = plt.subplots(nrows=1,ncols=2)
    ax = axs[0]
    br = make_br()
    bw = br.bw
    obs, beh = br.data_source.get_history()

    offset = 3
    current_location, current_behavior = br.data_source.get_pair(item="last_seen", offset=0)
    predicted_location, _ = br.data_source.get_pair(item="last_seen", offset=offset)

    ### good ###
    bpf.plot_2d(ax, obs, bw)
    bpf.plot_current_2d(ax, obs, bw)
    bpf.show_A(ax, bw)
    bpf.show_alpha(ax, br)
    bpf.show_behavior_variables(ax, br, beh)
    bpf.show_A_eigenspectrum(ax, bw)
    bpf.show_data_distance(ax, obs, max_step=50)

    bpf.plot_2d(axs[0], obs, bw)
    bpf.show_nstep_pred_pdf(axs[1], bw, axs[0], fig=fig, current_location=current_location, offset_location=predicted_location,  offset=offset)


    ### to fix ###
    # bpf.br_plot_3d(br)
    # bpf.show_Ct_y(ax, regressor=bw.regressor)
    # bpf.show_w_sideways(ax, bw, current_behavior)

def test_comparison_plots(make_br):
    brs = [make_br() for _ in range(3)]
    bpf.compare_metrics(brs, offset=0)