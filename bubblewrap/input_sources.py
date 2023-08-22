import numpy as np
import warnings

class NumpyDataSource:
    def __init__(self, obs, beh=None, time_offsets=()):
        self.obs = obs
        self.beh = beh
        self.time_offsets = time_offsets

        if beh is not None:
            assert len(self.beh) == len(self.obs)

        self.clear_range = (0, len(obs))
        if time_offsets:
            self.clear_range = (max(0, -min(time_offsets)), len(obs) - max(max(time_offsets), 0))

        self.index = 0


    def __len__(self):
        return self.clear_range[1] - self.clear_range[0]

    def get_pair_shapes(self):
        if self.beh is not None:
            return self.obs.shape[1], self.beh.shape[1]
        else:
            return self.obs.shape[1], 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            o, b = self.get_pair(self.index)
        except IndexError:
            raise StopIteration()

        offset_pairs = {}
        for offset in self.time_offsets:
            offset_pairs[offset] = self.get_pair(self.index, offset)

        self.index += 1
        return o, b, offset_pairs

    def get_pair(self, item, offset=0):
        # could be __getitem__

        if item == "last_seen":
            item = self.index - 1
        if item < 0:
            raise IndexError("Negative indexes are not supported.")

        if item >= len(self):
            raise IndexError("Index out of range.")

        inside_index = item + self.clear_range[0] + offset
        if self.beh is not None:
            return self.obs[inside_index,:], self.beh[inside_index,:]
        else:
            return self.obs[inside_index,:], None

    def get_history(self, depth=None):
        slice_end = self.index + self.clear_range[0]
        slice_start = self.clear_range[0]
        if depth is not None:
            slice_start = slice_end - depth

        if slice_start < self.clear_range[0]:
            raise IndexError()

        o = self.obs[slice_start:slice_end,:]
        b = None
        if self.beh is not None:
            b = self.obs[slice_start:slice_end, :]
        return o, b
