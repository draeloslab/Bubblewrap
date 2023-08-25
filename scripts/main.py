import numpy as np
from bubblewrap import Bubblewrap, BWRun, NumpyPairedDataSource, AnimationManager, default_clock_parameters, SymmetricNoisyRegressor
from bubblewrap.input_sources import HMM, HMMSimDataSource
import bubblewrap.plotting_functions as bpf

def example_movie():
    # define the data to feed into bubblewrap
    hmm = HMM.gaussian_clock_hmm(n_states=8,p1=.9)
    ds = HMMSimDataSource(hmm=hmm, seed=42, length=150, time_offsets=(1,))

    # define the bubblewrap object
    bw = Bubblewrap(dim=2, **default_clock_parameters)

    # define the (optional) method to regress the HMM state from `bw.alpha`
    reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=1, forgetting_factor=1 - (1e-2), noise_scale=1e-5)

    # define how we want to animate the progress
    class CustomAnimation(AnimationManager):
        def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
            historical_observations = br.data_source.get_history()[0]

            bpf.show_A(self.ax[0, 0], bw)
            bpf.show_A_eigenspectrum(self.ax[0, 1], bw)
            bpf.show_bubbles_2d(self.ax[1,0], historical_observations, bw)
            bpf.show_nstep_pred_pdf(self.ax[1,1], br, other_axis=self.ax[1,0], fig=self.fig, offset=1)
    am = CustomAnimation()

    # define the object to coordinate all the other objects
    br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, animation_manager=am)

    # run and save the output
    br.run()

def main():
    rng = np.random.default_rng()
    hmm = HMM.gaussian_clock_hmm(20, .5, variance_scale=3, radius=10)
    states, obs = hmm.simulate_with_states(1, rng)
    states, obs = hmm.simulate_with_states(1, rng, start_state=states[-1])
    print(obs)

if __name__ == '__main__':
    example_movie()