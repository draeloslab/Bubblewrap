import numpy as np
from bubblewrap import Bubblewrap, BWRun, NumpyPairedDataSource, AnimationManager, default_rwd_parameters, \
    SymmetricNoisyRegressor
from bubblewrap.regressions import NearestNeighborRegressor
from bubblewrap.input_sources.data_sources import HMMSimDataSourceSingle, NumpyDataSource, PairWrapperSource
import bubblewrap.input_sources.functional as fin

def get_data(file="monkey_reach_reduced.npz"):
    # monkey_reach_reduced.npz
    # jpca_reduced_sc.npz
    obs, beh = fin.get_from_saved_npz(file)

    concatenated = fin.zscore(np.hstack([obs, beh]))
    input_datasets = {
        'p(n)': fin.zscore(obs),
        'b': fin.zscore(beh),
        '[p(n), b]': concatenated,
        'p3([p(n), b])': fin.prosvd_data(concatenated, 3, 30),
        'p1([p(n), b])': fin.prosvd_data(concatenated, 1, 30),
        'p1(p(n))': fin.prosvd_data(fin.zscore(obs), 1, 30),
    }
    output_datasets = {
        'b': beh,
        'p1(p(n))': fin.prosvd_data(fin.zscore(obs), 1, 30),
        'p1([p(n), b])': fin.prosvd_data(concatenated, 1, 30),
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
            i = input_datasets[ikey]
            o  = output_datasets[okey]
            l = min(len(i), len(o))
            ds = PairWrapperSource(NumpyDataSource(i[:l], time_offsets=(1,5)), NumpyDataSource(o[:l], time_offsets=(1,5)))

            bw = Bubblewrap(dim=ds.output_shape[0], **default_rwd_parameters)
            reg = SymmetricNoisyRegressor(input_d=bw.N, output_d=ds.output_shape[1])
            br = BWRun(bw=bw, data_source=ds, behavior_regressor=reg, show_tqdm=True,
                       output_directory="/home/jgould/Documents/Bubblewrap/generated/bubblewrap_runs/")

            br.run(limit=10_000, )
            err = np.squeeze(br.behavior_error_history[1][-1000:])
            pred = np.squeeze(br.behavior_pred_history[1][-1000:])
            correct = pred - err
            print(f"{np.corrcoef(pred, correct)[0,1]}\t", end="")
        print()


if __name__ == '__main__':
    main()
