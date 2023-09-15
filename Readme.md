# Bubblewrap

## Quickstart

```python
from bubblewrap import Bubblewrap, BWRun, AnimationManager, default_clock_parameters, SymmetricNoisyRegressor
from bubblewrap.input_sources import HMM, HMMSimDataSourceSingle
import bubblewrap.plotting_functions as bpf

# define the data to feed into bubblewrap
hmm = HMM.gaussian_clock_hmm(n_states=8, p1=.9)
ds = HMMSimDataSourceSingle(hmm=hmm, seed=42, length=150, time_offsets=(1,))

# define the bubblewrap object
bw = Bubblewrap(dim=2, **default_clock_parameters)

# define the (optional) method to regress the HMM state from `bw.alpha`
reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=1)


# define how we want to animate the progress
class CustomAnimation(AnimationManager):
    # this function is called every step to add a frame
    def custom_draw_frame(self, step, bw: Bubblewrap, br: BWRun):
        historical_observations = br.data_source.get_history()[0]

        # show an image of the transition matrix in the top-left axis
        bpf.show_A(self.ax[0, 0], bw)

        # plot the eigenspectrum of the transition matrix on the top right
        bpf.show_A_eigenspectrum(self.ax[0, 1], bw)

        # show the locations of the bubbles on the bottom left
        bpf.show_bubbles_2d(self.ax[1, 0], historical_observations, bw)

        # show the estimated likelihood across space for the next data point
        bpf.show_nstep_pred_pdf(self.ax[1, 1], br, other_axis=self.ax[1, 0], fig=self.fig, offset=1)


# instantiate the (optional) object to make the video
am = CustomAnimation()

# define the object to coordinate all the other objects
br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, animation_manager=am)

# run and save the output
br.run()
```

## Requirements
Our algorithm is implemented in python with some extra packages including: numpy, jax (for GPU), and matplotlib (for plotting). 

We used python version 3.9 and conda to install the libraries listed in the environment file. 
We provide an environment file for use with conda to create a new environment with these requirements, though we note that jax requires additional setup for GPU integration (https://github.com/google/jax). 





## Refrence
The core Bubblewrap algorithm was initially described here: ['Bubblewrap: Online tiling and real-time flow prediction on neural manifolds'](https://proceedings.neurips.cc/paper/2021/hash/307eb8ee16198da891c521eca21464c1-Abstract.html).
