from bubblewrap.regressions import NearestNeighborRegressor, SymmetricNoisyRegressor, WindowRegressor
import pytest
import numpy as np

@pytest.fixture(params=["nearest_n", "noisy", "window"])
def reg_maker(request):
    if request.param == "nearest_n":
        return NearestNeighborRegressor
    elif request.param == "noisy":
        return SymmetricNoisyRegressor
    elif request.param == "window":
        return WindowRegressor

def test_can_run_1d(reg_maker, rng):
    w = np.array([2, -3])
    def f(point):
        return w @ point + 4
    space = np.linspace(0, 1, 100)

    reg = reg_maker(2, 1)
    for _ in range(1_000):
        x = rng.choice(space, size=2)
        y = f(x)
        pred = reg.predict(x)
        if not np.any(np.isnan(pred)):
            assert np.linalg.norm(pred - y) < 1e2
        reg.safe_observe(x=x, y=y)

def test_can_run_nd(reg_maker, rng):
    m, n = 4, 3
    w = rng.random(size=(m, n))
    def f(point):
        return w @ point + 4
    space = np.linspace(0, 1, 100)

    reg = reg_maker(n, m)
    for _ in range(1_000):
        x = rng.choice(space, size=n)
        y = f(x)
        pred = reg.predict(x)
        if not np.any(np.isnan(pred)):
            assert np.linalg.norm(pred - y) < 1e2
        reg.safe_observe(x=x, y=y)

def test_nan_at_right_time(reg_maker, rng):
    m, n = 4, 3
    space = np.linspace(0, 1, 100)

    w = rng.random(size=(m, n))
    def f(point):
        return w @ point + 4

    reg = reg_maker(n, m)
    assert np.all(np.isnan(reg.predict(rng.choice(space, size=n))))
    for _ in range(1_000):
        x = rng.choice(space, size=n)
        y = f(x)
        reg.safe_observe(x=x, y=y)

    x = rng.choice(space, size=n)
    assert np.all(~np.isnan(reg.predict(x)))
