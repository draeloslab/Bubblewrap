import numpy as np
import abc

# s = np.load(file)
#
# if "npy" in file:
#     data = s.T
# elif "npz" in file:
#     data = s['y'][0]
#     pre_obs = s['x']
#
#     data = np.tile(data, reps=(tiles, 1))
#     obs = pre_obs
#     for i in range(1, tiles):
#         c = (-1) ** i if invert_alternate_behavior else 1
#         obs = np.hstack((obs, pre_obs * c))
#
#
#
# obs = obs.reshape((-1,1))
# old_data = np.array(data)
# old_obs = np.array(obs)
#
# if data_transform == "n,b":
#     data = np.hstack([data, obs])
# elif data_transform == "n":
#     data = np.hstack([data])
# elif data_transform == "b":
#     data = np.hstack([obs])
# else:
#     raise Exception("You need to set data_transform")
#
#
# obs = np.hstack([old_data,old_obs])

class NumpyDataSource:
    def __init__(self, obs, beh, time_offsets=()):
        self.obs = obs
        self.beh = beh
        assert len(self.beh) == len(self.obs)
        self.time_offsets = time_offsets

        self.clear_range = (0, len(obs))
        if time_offsets:
            self.clear_range = (max(0, -min(time_offsets)), len(obs) - max(max(time_offsets), 0))

    def __len__(self):
        return self.clear_range[1] - self.clear_range[0]

    def __iter__(self):
        for i in range(*self.clear_range):
            o, b = self.obs[i,:], self.beh[i,:]

            offset_pairs = {}
            for offset in self.time_offsets:
                offset_pairs[offset] = (self.obs[i,:], self.beh[i,:])

            yield o, b, offset_pairs

    def get_pair_shapes(self):
        return self.obs.shape[1], self.beh.shape[1]