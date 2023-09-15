import numpy as np
from bubblewrap import Bubblewrap, BWRun, AnimationManager, default_clock_parameters, default_rwd_parameters, \
    SymmetricNoisyRegressor
from bubblewrap.input_sources.data_sources import NumpyDataSource, PairWrapperSource
from bubblewrap.input_sources import HMM, HMMSimDataSourceSingle
import bubblewrap.plotting_functions as bpf
from optim import evaluate


def example_movie():
    # define the data to feed into bubblewrap
    hmm = HMM.gaussian_clock_hmm(n_states=8, p1=.9)
    ds = HMMSimDataSourceSingle(hmm=hmm, seed=42, length=150, time_offsets=(1,))

    # define the bubblewrap object
    bw = Bubblewrap(dim=6, **default_clock_parameters)

    # define the (optional) method to regress the HMM state from `bw.alpha`
    reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=1)

    # define how we want to animate the progress
    class CustomAnimation(AnimationManager):
        def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
            historical_observations = br.data_source.get_history()[0]

            bpf.show_A(self.ax[0, 0], bw)
            bpf.show_A_eigenspectrum(self.ax[0, 1], bw)
            bpf.show_bubbles_2d(self.ax[1, 0], historical_observations, bw)
            bpf.show_nstep_pred_pdf(self.ax[1, 1], br, other_axis=self.ax[1, 0], fig=self.fig, offset=1)

    am = CustomAnimation()

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, animation_manager=am,
               output_directory="../generated/bubblewrap_runs/")

    # run and save the output
    br.run()


def main():
    evaluate({"B_thresh": 1})


if __name__ == '__main__':
    main()
