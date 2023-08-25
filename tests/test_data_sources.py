import numpy as np

from bubblewrap.input_sources import NumpyPairedDataSource, HMMSimDataSource, HMM, ProSVDDataSource
from bubblewrap.input_sources.data_sources import PairedDataSource, PairWrapperSource, NumpyDataSource
import pytest


@pytest.fixture(params=["numpy", "hmm", "pro"])
def ds(rng, request):
    m = 500
    if request.param == "numpy":
        t = np.linspace(0,30*np.pi, m)
        obs = np.vstack([np.sin(t), np.cos(t)]).T
        return NumpyPairedDataSource(obs, np.mod(t, np.pi), time_offsets=(-10, -1, 0, 10))

    elif request.param == "hmm":
        hmm = HMM.gaussian_clock_hmm(20, p1=.9, angle=0, variance_scale=2, radius=10)
        return HMMSimDataSource(hmm, seed=0, length=m, time_offsets=(-10, -1, 0, 10))

    elif request.param == "pro":
        init_size = 100
        t = np.linspace(0,30*np.pi, m + init_size)
        obs = np.vstack([np.sin(t), np.cos(t)]).T
        ds = ProSVDDataSource(input_source=NumpyDataSource(obs), output_d=1, init_size=init_size, time_offsets=(-10, -1, 0, 10))

        return PairWrapperSource(NumpyDataSource(obs), ds)

# todo: test with no time_offset

def test_can_run(ds):
    ds: PairedDataSource
    ds.get_pair_shapes()
    for _ in ds:
        for t in ds.time_offsets:
            ds.get_atemporal_data_pair(offset=t)
        ds.get_history()

def test_history_is_correct(ds):
    ds: PairedDataSource
    last_obs, last_beh = None, None
    for idx, (obs, beh, offset_pairs) in enumerate(ds):
        if idx > 0:
            historical_obs, historical_beh = ds.get_atemporal_data_pair(offset = -1)
            assert np.all(last_obs == historical_obs)
            assert np.all(last_beh == historical_beh)

            historical_obs, historical_beh = offset_pairs[-1]
            assert np.all(last_obs == historical_obs)
            assert np.all(last_beh == historical_beh)

            curr_obs, curr_beh = ds.get_atemporal_data_pair(offset = 0)
            assert np.all(obs == curr_obs)
            assert np.all(beh == curr_beh)
        last_obs = obs
        last_beh = beh

def test_reusable(ds):
    pass

def test_serializable(ds):
    pass
