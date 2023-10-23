import numpy as np
import bubblewrap.input_sources.functional as fin
import bubblewrap as bw

def main2():
    obs, beh = fin.get_from_saved_npz("indy_full.npz")

    datasets = {
        'a': obs,
        'b': fin.prosvd_data(obs, 30, 30),
        'c': fin.prosvd_data(obs, 3, 30),
        'd': fin.prosvd_data(obs, 1, 30),
        'e': beh,
        'f': fin.zscore(beh, 30),
        'g': fin.prosvd_data(fin.zscore(beh, 10), 1, 20),
        'h': fin.prosvd_data(np.hstack([obs[10:], fin.zscore(beh, 10)]), 5, 20),
    }
    keys = list(datasets.keys())

    input_keys = 'b c d f g h'.split(" ")
    output_keys = 'd h'.split(" ")

    results = {}
    true_values = {}
    for okey in output_keys:
        results[okey] = {}
        for ikey in input_keys:
            print(f"{okey= } {ikey= }")
            i, o = fin.clip(datasets[ikey], datasets[okey])

            alpha_dict = fin.bwrap_alphas_ahead(input_arr=i, bw_params= bw.default_parameters.default_jpca_dataset_parameters, nsteps=[0,1])

            a_current, a_ahead, o = fin.clip(alpha_dict[0], alpha_dict[1], o)

            reg = bw.SymmetricNoisyRegressor(input_d=a_current.shape[1], output_d=o.shape[1], init_min_ratio=5)

            pred = []

            for i, (x, y) in enumerate(list(zip(a_current, o))[:-1]):
                reg.safe_observe(x, y)
                pred.append(reg.predict(a_ahead[i]))
            results[okey][ikey] = np.array(pred)

            if okey not in true_values:
                true_values[okey] = o[1:]

            # err = np.squeeze(br.behavior_error_history[1][-1000:])
            # pred = np.squeeze(br.behavior_pred_history[1][-1000:])
            # correct = pred - err

def main():
    from bubblewrap.input_sources.functional import make_hashable
    import hashlib

    obs, beh = fin.get_from_saved_npz("indy_full.npz")
    obs = np.eye(2)
    # b = fin.prosvd_data(obs, 30, 30)
    # d = fin.prosvd_data(obs, 1, 30)

    # i, o = b, d

    print()

if __name__ == '__main__':
    main()