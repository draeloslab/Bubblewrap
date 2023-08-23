import numpy as np
import warnings
from collections import deque

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hmm_simulation import HMM

class DataSource:
    def __init__(self, time_offsets=()):
        self.time_offsets = time_offsets

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def get_pair_shapes(self):
        pass

    def get_atemporal_data_pair(self, offset=0):
        pass

    def get_history(self, depth=None):
        pass

class HMMSimDataSource(DataSource):
    def __init__(self, hmm, seed, length, time_offsets=(), min_memory_radius=200):
        super().__init__(time_offsets)
        self.hmm: HMM = hmm
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.necessary_buffer = [-1*min(min(time_offsets),0) + 1, max(max(time_offsets),0)] # +1 for current state
        self.memory_radius = max(min_memory_radius, 2 * max([abs(t) for t in time_offsets]))
        self.length = length

        self.future = deque(maxlen=self.memory_radius + self.necessary_buffer[1])
        self.past = deque(maxlen=self.memory_radius + self.necessary_buffer[0])
        # present is past[0], 1 step in the future is future[0]

        beh, obs = self.hmm.simulate_with_states(self.necessary_buffer[0], self.rng)
        for i in range(obs.shape[0]):
            self.past.appendleft((obs[i], beh[i]))
        self.current_state = beh[-1]

        self.simulate_more_steps()
        self.index = None

    def simulate_more_steps(self):
        n = self.memory_radius - len(self.future)
        beh, obs = self.hmm.simulate_with_states(n, self.rng, self.current_state)
        self.current_state = beh[-1]

        for i in range(obs.shape[0]):
            self.future.append((obs[i], beh[i]))



    def __len__(self):
        return self.length

    def get_pair_shapes(self):
        return self.hmm.emission_model.embedded_dimension, 1

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration()

        current = self.future.popleft()
        self.past.appendleft(current)
        # todo: worry about references screwing with stuff?
        if len(self.future) < self.necessary_buffer[1]:
            self.simulate_more_steps()

        offset_pairs = {}
        for offset in self.time_offsets:
            offset_pairs[offset] = self.get_atemporal_data_pair(offset)

        self.index += 1
        return current[0], current[1], offset_pairs

    def get_atemporal_data_pair(self, offset=0):
        if offset <= 0:
            return self.past[-offset]
        if offset > 0:
            return self.future[offset - 1]


    def get_history(self, depth=None):
        obs, beh = tuple(zip(*self.past))
        return np.array(obs), np.array(beh)

class NumpyDataSource(DataSource):
    def __init__(self, obs, beh=None, time_offsets=()):
        super().__init__(time_offsets)

        self.obs = obs
        self.beh = beh
        if beh is not None and len(beh.shape) == 1:
                self.beh = beh[:,None]

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
            o, b = self._get_pair(self.index, offset=0)
        except IndexError:
            raise StopIteration()

        offset_pairs = {}
        for offset in self.time_offsets:
            offset_pairs[offset] = self._get_pair(item=self.index, offset=offset)

        self.index += 1
        return o, b, offset_pairs

    def get_atemporal_data_pair(self, offset=0):
        """gets a data pair relative to the present pair"""
        return self._get_pair(item=self.index - 1, offset=offset)

    def _get_pair(self, item, offset=0):
        # could be __getitem__

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
