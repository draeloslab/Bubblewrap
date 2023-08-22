import numpy as np
from bubblewrap import Bubblewrap, BWRun, NumpyDataSource
from bubblewrap.bw_run import AnimationManager
from bubblewrap.default_parameters import default_clock_parameters
import bubblewrap.plotting_functions as bpf


def main():
    rng = np.random.default_rng()
    m, n_obs, n_beh = 200, 3, 4
    obs = rng.normal(size=(m, n_obs))
    ds = NumpyDataSource(obs, time_offsets=(-10, 0, 10))

    bw = Bubblewrap(3, **default_clock_parameters)

    class SimpleAnimation(AnimationManager):
        def custom_draw_frame(self, step, bw, br):
            bpf.show_alpha(self.ax[0, 0], br)
            bpf.show_A(self.ax[0, 1], bw)
            bpf.show_A_eigenspectrum(self.ax[1, 0], bw)
            # bpf.show_Ct_y(self.ax[1,1], bw)
    am = SimpleAnimation()
    br = BWRun(bw, ds, am)
    br.run()

if __name__ == '__main__':
    main()