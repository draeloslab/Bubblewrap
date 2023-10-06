import numpy as np
import os
import pickle
import warnings
import json
from proSVD import proSVD
import bubblewrap

TURN_OFF_CACHING = False
if TURN_OFF_CACHING:
    warnings.warn("Caching is turned off.")


dataset_base_path = "/home/jgould/Documents/Bubblewrap/generated/datasets/"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def make_hashable(x):
    return json.dumps(x, sort_keys=True, cls=NumpyEncoder).encode()

def save_to_cache(file, location=os.path.join(dataset_base_path, "cache")):
    cache_index_file = os.path.join(location, f"{file}_index.pickle")
    try:
        with open(cache_index_file, 'rb') as fhan:
            cache_index = pickle.load(fhan)
    except FileNotFoundError:
        cache_index = {}

    def decorator(original_function):
        if TURN_OFF_CACHING:
            return original_function

        def new_function(**kwargs):
            kwargs_as_key = make_hashable(kwargs)
            if kwargs_as_key not in cache_index:
                result = original_function(**kwargs)

                hstring = str(hash(make_hashable(result)))[-15:]
                cache_file = os.path.join(location,f"{file}_{hstring}.pickle")
                with open(cache_file, "wb") as fhan:
                    pickle.dump(result, fhan)

                cache_index[kwargs_as_key] = cache_file
                with open(cache_index_file, 'bw') as fhan:
                    pickle.dump(cache_index, fhan)

            with open(os.path.join(location, cache_index[kwargs_as_key]), 'rb') as fhan:
                return pickle.load(fhan)

        return new_function
    return decorator


def get_from_saved_npz(filename):
    dataset = np.load(os.path.join(dataset_base_path, filename))
    beh = dataset['x']

    if len(dataset['y'].shape) == 3:
        obs = dataset['y'][0]
    else:
        obs = dataset['y']

    return obs, beh.reshape([obs.shape[0], -1])


def prosvd_data(input_arr, output_d, init_size):
    pro = proSVD(k=output_d)
    pro.initialize(input_arr[:init_size].T)

    output = []
    for i in range(init_size, len(input_arr)):
        obs = input_arr[i:i + 1, :]
        pro.preupdate()
        pro.updateSVD(obs.T)
        pro.postupdate()

        obs = obs @ pro.Q

        output.append(obs)
    return np.array(output).reshape((-1, output_d))


def zscore(input_arr, init_size=6):
    output = []
    for i in range(len(input_arr)):
        if i > init_size:
            output.append((input_arr[i] - input_arr[:i].mean(axis=0)) / input_arr[:i].std(ddof=1, axis=0))
    return np.array(output)

@save_to_cache("bwrap_alphas")
def bwrap_alphas(input_arr, bw_params):
    alphas = []
    bw = bubblewrap.Bubblewrap(dim=input_arr.shape[1], **bw_params)
    for step in range(len(input_arr)):
        bw.observe(input_arr[step])

        if step < bw.M:
            pass
        elif step == bw.M:
            bw.init_nodes()
            bw.e_step()
            bw.grad_Q()
        else:
            bw.e_step()
            bw.grad_Q()

            alphas.append(bw.alpha)
    return np.array(alphas)

@save_to_cache("bwrap_alphas_ahead")
def bwrap_alphas_ahead(input_arr, bw_params, nsteps=1):
    alphas = []
    bw = bubblewrap.Bubblewrap(dim=input_arr.shape[1], **bw_params)
    for step in range(len(input_arr)):
        bw.observe(input_arr[step])

        if step < bw.M:
            pass
        elif step == bw.M:
            bw.init_nodes()
            bw.e_step()
            bw.grad_Q()
        else:
            bw.e_step()
            bw.grad_Q()

            alphas.append(bw.alpha @ np.linalg.matrix_power(bw.A, nsteps))
    return np.array(alphas)


def main():
    obs, beh = get_from_saved_npz("jpca_reduced_sc.npz")
    obs = zscore(prosvd_data(obs, output_d=2, init_size=30), init_size=3)
    alphas = bwrap_alphas(input_arr=obs, bw_params=bubblewrap.default_parameters.default_jpca_dataset_parameters)
    print(alphas)


if __name__ == '__main__':
    main()