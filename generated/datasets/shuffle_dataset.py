import sys
import numpy as np

def make_shuffled_file(file):
    if "/" in file:
        raise Exception("no paths, just files")

    new_file = "shuffled_" + file

    rng = np.random.default_rng(42)

    s = np.load(file)
    if "npy" in file:
        data = s.T
        permutation = rng.permutation(np.arange(data.shape[0]))
        data = data[permutation]
        data = data.T
        np.save(new_file, data)
    elif "npz" in file:
        y = s['y']
        x = s['x']

        if len(x.shape) != 1:
            raise Exception()

        permutation = rng.permutation(np.arange(y.shape[1]))
        y = y[:, permutation, :]
        x = x[permutation]
        np.savez(new_file, y=y, x=x)
    else:
        raise Exception("Unknown filetype")

if __name__ == '__main__':
    file = sys.argv[1]
    make_shuffled_file(file)
