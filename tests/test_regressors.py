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

def test_can_run(reg_maker, rng):
    def f(point):
        return np.array([2, -3]) @ point + 4
    space = np.linspace(0, 1, 100)

    reg = reg_maker(2, 1)
    for _ in range(1_000):
        x = rng.choice(space, size=2)
        y = f(x)
        pred = reg.predict(x)
        if not np.any(np.isnan(pred)):
            assert np.linalg.norm(pred - y) < 1e2
        reg.safe_observe(x=x, y=y)