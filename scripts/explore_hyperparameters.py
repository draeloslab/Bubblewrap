import numpy as np
import time
from bubblewrap_run import BubblewrapRun
from run_bubblewrap import default_parameters

def generate_random_bw_hyperparameters(variable_parameters=None):
    rng = np.random.default_rng()

    if variable_parameters is None:
        variable_parameters = dict(
            num=[16, 256, 1024],
            lam=[1e-4, 1e-3, 1e-3, 1e-3, 1e-2],
            nu=[1e-4, 1e-3, 1e-3, 1e-3, 1e-2],
            eps=[1e-4, 1e-3, 1e-2],
            B_thresh=[-15, -10, -5],
            seed=[10 * x for x in range(100)]
        )
    while True:
        d = dict(default_parameters)
        for key, value in variable_parameters.items():
            d[key] = rng.choice(value)
        f = ...
        yield d, f


def do_many_random_runs():
    for p, f in generate_random_bw_hyperparameters():
        try:
            start_time = time.time()
            bw = run_bubblewrap(default_file, p)
            end_time = time.time()
            br = BubblewrapRun(bw, file=f, bw_parameters=p, time_to_run=end_time-start_time)
            br.save()
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
