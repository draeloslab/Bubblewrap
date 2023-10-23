import numpy as np
import os
import pickle
import json
from proSVD import proSVD
import bubblewrap
from tqdm import tqdm
import hashlib



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def make_hashable(x):
    return json.dumps(x, sort_keys=True, cls=NumpyEncoder).encode()


def save_to_cache(file, location=os.path.join(bubblewrap.config.CONFIG["data_path"], "cache")):
    if not os.path.exists(location):
        os.mkdir(location)
    cache_index_file = os.path.join(location, f"{file}_index.pickle")
    try:
        with open(cache_index_file, 'rb') as fhan:
            cache_index = pickle.load(fhan)
    except FileNotFoundError:
        cache_index = {}

    def decorator(original_function):
        if not bubblewrap.config.CONFIG["attempt_to_cache"]:
            return original_function

        def new_function(**kwargs):
            kwargs_as_key = int(hashlib.sha1(make_hashable(kwargs)).hexdigest(), 16)

            if kwargs_as_key not in cache_index:
                result = original_function(**kwargs)

                hstring = str(kwargs_as_key)[-15:]
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
    dataset = np.load(os.path.join(bubblewrap.config.CONFIG["data_path"], filename))
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
    for i in tqdm(range(init_size, len(input_arr))):
        obs = input_arr[i:i + 1, :]
        pro.preupdate()
        pro.updateSVD(obs.T)
        pro.postupdate()

        obs = obs @ pro.Q

        output.append(obs)
    return np.array(output).reshape((-1, output_d))


def zscore(input_arr, init_size=6):
    mean = 0
    m2 = 1e-4
    output = []
    for i, x in enumerate(tqdm(input_arr)):
        if i >= init_size:
            std = np.sqrt(m2 / (i - 1))
            output.append((x - mean) / std)

        delta = x - mean
        mean += delta / (i + 1)
        m2 += delta * (x - mean)

    return np.array(output)


def shuffle_time(input_arr_list, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p = rng.permutation(input_arr_list[0].shape[0])

    return (x[p,:] for x in input_arr_list)


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
def bwrap_alphas_ahead(input_arr, bw_params, nsteps=(1,)):
    returns = {x:[] for x in nsteps}
    bw = bubblewrap.Bubblewrap(dim=input_arr.shape[1], **bw_params)
    for step in tqdm(range(len(input_arr))):
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

            for step in nsteps:
                returns[step].append(bw.alpha @ np.linalg.matrix_power(bw.A, step))
    returns = {x: np.array(returns[x]) for x in returns}
    return returns

def clip(*args):
    l = min([len(a) for a in args])
    args = [a[-l:] for a in args]

    l = max([np.nonzero(np.all(np.isfinite(a), axis=1))[0][0] for a in args])
    args = [a[l:] for a in args]
    return args


def main():
    obs, beh = get_from_saved_npz("jpca_reduced_sc.npz")
    obs = zscore(prosvd_data(obs, output_d=2, init_size=30), init_size=3)
    alphas = bwrap_alphas(input_arr=obs, bw_params=bubblewrap.default_parameters.default_jpca_dataset_parameters)
    print(alphas)


if __name__ == '__main__':
    main()