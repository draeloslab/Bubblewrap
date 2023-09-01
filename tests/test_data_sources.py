import numpy as np

from bubblewrap.input_sources import NumpyPairedDataSource, HMMSimDataSource, HMM, ProSVDDataSource
from bubblewrap.input_sources.data_sources import ConsumableDataSource, PairWrapperSource, NumpyDataSource, ConcatenatorSource
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
        a = NumpyDataSource(obs, time_offsets=(-10, -1, 0, 10))
        b = ProSVDDataSource(input_source=NumpyDataSource(obs), output_d=1, init_size=init_size, time_offsets=a.time_offsets)
        return PairWrapperSource(b, a)

# todo: test with no time_offset

def test_can_run(ds):
    ds: ConsumableDataSource
    assert hasattr(ds, "output_shape")
    assert type(ds.output_shape) == tuple
    for idx, _ in enumerate(ds.triples(1e4)):
        assert idx < 600
        # assert idx < 500
        for t in ds.time_offsets:
            ds.get_atemporal_data_point(offset=t)
        ds.get_history()

    # assert idx == 499

def test_history_is_correct(ds):
    ds: ConsumableDataSource
    last_obs, last_beh = None, None
    for idx, (obs, beh, offset_pairs) in enumerate(ds.triples()):
        if idx > 0:
            historical_obs, historical_beh = ds.get_atemporal_data_point(offset = -1)
            assert np.all(last_obs == historical_obs)
            assert np.all(last_beh == historical_beh)

            historical_obs, historical_beh = offset_pairs[-1]
            assert np.all(last_obs == historical_obs)
            assert np.all(last_beh == historical_beh)

            curr_obs, curr_beh = ds.get_atemporal_data_point(offset = 0)
            assert np.all(obs == curr_obs)
            assert np.all(beh == curr_beh)
        last_obs = obs
        last_beh = beh


def test_prosvd_synced():
    arr = np.array([1,2,3,4,5])[:,None]
    a = NumpyDataSource(arr, time_offsets=())
    b = ProSVDDataSource(input_source=NumpyDataSource(arr), output_d=1, init_size=3, time_offsets=a.time_offsets)
    p = PairWrapperSource(b, a)
    aa, bb = next(p)
    assert np.allclose(aa,bb)

def test_can_load_file():
    obs, beh = NumpyDataSource.get_from_saved_npz("jpca_reduced.npz")

def test_can_concatenate():
    a = NumpyDataSource([1,2,3])
    b = NumpyDataSource([4,5,6])
    c = NumpyDataSource([7,8,9])
    d = ConcatenatorSource([a,b])
    ds = PairWrapperSource(c,d)
    n = next(ds)
    assert np.all(n[0] == np.array([7]))
    assert np.all(n[1] == np.array([1,4]))

def test_reusable(ds):
    pass

def test_serializable(ds):
    pass
