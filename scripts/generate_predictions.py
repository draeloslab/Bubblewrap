import numpy as np
from bubblewrap import Bubblewrap, BWRun, NumpyPairedDataSource, AnimationManager, default_rwd_parameters, \
    SymmetricNoisyRegressor
from bubblewrap.regressions import NearestNeighborRegressor
from bubblewrap.input_sources.data_sources import HMMSimDataSourceSingle, NumpyDataSource, PairWrapperSource, \
    ConcatenatorSource, ProSVDDataSourceSingle
import bubblewrap.plotting_functions as bpf
from copy import deepcopy

def get_data(file="jpca_reduced_sc.npz"):
    # monkey_reach_reduced.npz
    # jpca_reduced_sc.npz
    obs, beh = NumpyDataSource.get_from_saved_npz(file, time_offsets=(0, 1, 5))

    input_datasets = {
        'p(n)': obs,
        'b': beh,
        '[p(n), b]': ConcatenatorSource([obs, beh]),
        'p3([p(n), b])': ProSVDDataSourceSingle(ConcatenatorSource([obs.drop_time_offsets(), beh.drop_time_offsets()]),
                                                output_d=3, time_offsets=beh.time_offsets),
        'p1([p(n), b])': ProSVDDataSourceSingle(ConcatenatorSource([obs.drop_time_offsets(), beh.drop_time_offsets()]),
                                                output_d=1, time_offsets=beh.time_offsets),
        'p1(p(n))': ProSVDDataSourceSingle(obs.drop_time_offsets(), output_d=1, time_offsets=beh.time_offsets),
    }
    output_datasets = {
        'b': beh,
        'p1([p(n), b])': ProSVDDataSourceSingle(ConcatenatorSource([obs.drop_time_offsets(), beh.drop_time_offsets()]),
                                                 output_d=1, time_offsets=beh.time_offsets),
        'p1(p(n))': ProSVDDataSourceSingle(obs.drop_time_offsets(), output_d=1, time_offsets=beh.time_offsets),
    }
    return input_datasets, output_datasets


def main():
    input_keys = ['p(n)', 'b', '[p(n), b]', 'p3([p(n), b])', 'p1([p(n), b])', 'p1(p(n))']
    output_keys = ['b', 'p1(p(n))', 'p1([p(n), b])']

    input_datasets, output_datasets = get_data()
    assert set(input_keys) == set(input_datasets.keys())
    assert set(output_keys) == set(output_datasets.keys())

    for ikey in input_keys:
        for okey in output_keys:
            ds = PairWrapperSource(deepcopy(input_datasets[ikey]), deepcopy(output_datasets[okey]))
            bw = Bubblewrap(dim=ds.output_shape[0], **default_rwd_parameters)
            # reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=ds.output_shape[1])
            reg = NearestNeighborRegressor(input_d=bw.N, output_d=ds.output_shape[1])
            br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, show_tqdm=True,
                       output_directory="/home/jgould/Documents/Bubblewrap/generated/bubblewrap_runs/")

            br.run(limit=10_000, )
            err = np.squeeze(br.behavior_error_history[1][-1000:])
            pred = np.squeeze(br.behavior_pred_history[1][-1000:])
            correct = pred - err
            # print(f"{(br.behavior_error_history[1][-1000:] ** 2).mean()}\t", end=None)
            print(f"{np.corrcoef(pred, correct)[0,1]}\t", end="")
        print()


if __name__ == '__main__':
    main()
    # import pickle
    # with open('/home/jgould/Documents/Bubblewrap/generated/bubblewrap_runs/bubblewrap_run_2023-08-31-18-03-47.pickle', 'rb') as fhan:
    #     br = pickle.load(fhan)
    #     br
