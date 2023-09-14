import numpy as np
from bubblewrap import Bubblewrap, BWRun, NumpyPairedDataSource, AnimationManager, default_rwd_parameters, \
    SymmetricNoisyRegressor
from bubblewrap.regressions import NearestNeighborRegressor
from bubblewrap.input_sources.data_sources import HMMSimDataSourceSingle, NumpyDataSource, PairWrapperSource, \
    ConcatenatorSource, ProSVDDataSourceSingle
import bubblewrap.plotting_functions as bpf
from copy import deepcopy


def main():
    for i in range(6):
        obs, beh = NumpyDataSource.get_from_saved_npz("monkey_reach_reduced.npz", time_offsets=(0, 1, 5))
        p_beh = ProSVDDataSourceSingle(beh.drop_time_offsets(), output_d=1, time_offsets=beh.time_offsets)
        c_both = ConcatenatorSource([obs, beh], )
        p3_both = ProSVDDataSourceSingle(ConcatenatorSource([obs.drop_time_offsets(), beh.drop_time_offsets()]),
                                         output_d=3, time_offsets=beh.time_offsets)
        p1_both = ProSVDDataSourceSingle(ConcatenatorSource([obs.drop_time_offsets(), beh.drop_time_offsets()]),
                                         output_d=1, time_offsets=beh.time_offsets)

        if i == 0:
            ds = PairWrapperSource(obs, p_beh)
        elif i == 1:
            ds = PairWrapperSource(beh, p_beh)
        elif i == 2:
            ds = PairWrapperSource(c_both, p_beh)
        elif i == 3:
            ds = PairWrapperSource(p3_both, p_beh)
        elif i == 4:
            ds = PairWrapperSource(c_both, p1_both)
        elif i == 5:
            ds = PairWrapperSource(p3_both, p1_both)
        else:
            raise Exception()

        # define the bubblewrap object
        bw = Bubblewrap(dim=ds.output_shape[0], **default_rwd_parameters)

        # define the (optional) method to regress the HMM state from `bw.alpha`
        reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=ds.output_shape[1], forgetting_factor=1 - (1e-2),
                                      noise_scale=1e-5)
        # reg = NearestNeighborRegressor(input_d=bw.N, output_d=ds.output_shape[1])

        # define the object to coordinate all the other objects
        br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, show_tqdm=False,
                   output_directory="/home/jgould/Documents/Bubblewrap/generated/bubblewrap_runs/")

        # run and save the output
        br.run(limit=10_000)
        print(f"{i}: {(br.behavior_error_history[1][-1000:] ** 2).mean()} ({reg})")


if __name__ == '__main__':
    main()
    # import pickle
    # with open('/home/jgould/Documents/Bubblewrap/generated/bubblewrap_runs/bubblewrap_run_2023-08-31-18-03-47.pickle', 'rb') as fhan:
    #     br = pickle.load(fhan)
    #     br
