import numpy as np
from collections import deque
from proSVD import proSVD
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hmm_simulation import HMM


class PairedDataSource(ABC):
    def __init__(self, time_offsets=()):
        self.time_offsets = time_offsets

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_pair_shapes(self):
        pass

    @abstractmethod
    def get_atemporal_data_pair(self, offset=0):
        pass

    @abstractmethod
    def get_history(self, depth=None):
        pass

class DataSource(ABC):
    def __init__(self, shape, time_offsets=()):
        self.time_offsets = time_offsets
        self.shape = shape

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_atemporal_data_point(self, offset=0):
        pass

    @abstractmethod
    def get_history(self, depth=None):
        pass


class PairWrapperSource(PairedDataSource):
    def __init__(self, obs, beh, time_offsets=(), default_length=1_000):
        super().__init__(time_offsets=time_offsets)
        self.obs: DataSource = obs
        self.beh: DataSource = beh

        self.length = default_length
        if hasattr(obs, "__len__"):
            assert len(obs) == len(beh)
            self.length = len(obs)

        self.index = None

    def __len__(self):
        return self.length

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        pairs = {}
        for offset in self.time_offsets:
            pairs[offset] = self.get_atemporal_data_pair(offset)
        o = next(self.obs)
        b = next(self.beh)
        return o, b, pairs

    def get_pair_shapes(self):
        return self.obs.shape, self.beh.shape

    def get_atemporal_data_pair(self, offset=0):
        return self.obs.get_atemporal_data_point(offset), self.beh.get_atemporal_data_point(offset)

    def get_history(self, depth=None):
        return self.obs.get_history(), self.beh.get_history()


class StreamDataSource(DataSource, ABC):
    def __init__(self, shape, time_offsets=(), min_memory_radius=500):
        super().__init__(shape, time_offsets)

        self.necessary_buffer = [-1*min(min(time_offsets, default=0),0) + 1, max(max(time_offsets, default=0),0)] # +1 for current state
        self.memory_radius = max(min_memory_radius, 2 * max([abs(t) for t in time_offsets], default=0))

        self.future = deque(maxlen=self.memory_radius + self.necessary_buffer[1])
        self.past = deque(maxlen=self.memory_radius + self.necessary_buffer[0])
        # present is past[0], 1 step in the future is future[0]

    @abstractmethod
    def simulate_more_steps(self):
        pass

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        current = self.future.popleft()
        self.past.appendleft(current)

        if len(self.future) < self.necessary_buffer[1]:
            self.simulate_more_steps()

        self.index += 1
        return current

    def get_atemporal_data_point(self, offset=0):
        if offset <= 0:
            return self.past[-offset]
        if offset > 0:
            return self.future[offset - 1]

    def get_history(self, depth=None):
        return np.array(self.past)


class HMMSimDataSource(PairedDataSource):
    def __init__(self, hmm, seed, length, time_offsets=(), min_memory_radius=1000):
        super().__init__(time_offsets=time_offsets)
        self.necessary_buffer = [-1*min(min(time_offsets, default=0),0) + 1, max(max(time_offsets, default=0),0)] # +1 for current state
        self.memory_radius = max(min_memory_radius, 2 * max([abs(t) for t in time_offsets], default=0))

        self.future = deque(maxlen=self.memory_radius + self.necessary_buffer[1])
        self.past = deque(maxlen=self.memory_radius + self.necessary_buffer[0])
        # present is past[0], 1 step in the future is future[0]

        self.index = None


        self.hmm: HMM = hmm
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.length = length

        beh, obs = self.hmm.simulate_with_states(self.necessary_buffer[0], self.rng)
        for i in range(obs.shape[0]):
            self.past.appendleft((obs[i], beh[i]))
        self.current_state = beh[-1]

        self.simulate_more_steps()

    def simulate_more_steps(self):
        n = self.memory_radius - len(self.future)
        beh, obs = self.hmm.simulate_with_states(n, self.rng, self.current_state)
        self.current_state = beh[-1]

        for i in range(obs.shape[0]):
            self.future.append((obs[i], beh[i]))


    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        current = self.future.popleft()
        self.past.appendleft(current)

        if len(self.future) < self.necessary_buffer[1]:
            self.simulate_more_steps()

        self.index += 1
        return current

    def get_atemporal_data_pair(self, offset=0):
        if offset <= 0:
            return self.past[-offset]
        if offset > 0:
            return self.future[offset - 1]

    def get_history(self, depth=None):
        return np.array(self.past)

    def __len__(self):
        return self.length

    def get_pair_shapes(self):
        return self.hmm.emission_model.embedded_dimension, 1


class NumpyDataSource(DataSource):
    def __init__(self, a, time_offsets=()):
        super().__init__(a.shape[1], time_offsets)
        self.a = a

        self.clear_range = (0, a.shape[0])
        if time_offsets:
            self.clear_range = (max(0, -min(time_offsets)), len(a) - max(max(time_offsets), 0))


    def __len__(self):
        return self.clear_range[1] - self.clear_range[0]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            d = self._get_item(self.index, offset=0)
        except IndexError:
            raise StopIteration()

        self.index += 1
        return d
    #
    def get_atemporal_data_point(self, offset=0):
        """gets a data pair relative to the present pair"""
        return self._get_item(item=self.index - 1, offset=offset)
    #
    def _get_item(self, item, offset=0):
        if item < 0:
            raise IndexError("Negative indexes are not supported.")

        if item >= len(self):
            raise IndexError("Index out of range.")

        inside_index = item + self.clear_range[0] + offset
        return self.a[inside_index]

    def get_history(self, depth=None):
        slice_end = self.index + self.clear_range[0]
        slice_start = self.clear_range[0]
        if depth is not None:
            slice_start = slice_end - depth

        if slice_start < self.clear_range[0]:
            raise IndexError()

        return self.a[slice_start:slice_end]


class NumpyPairedDataSource(PairedDataSource):
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


class ProSVDDataSource(StreamDataSource):
    def __init__(self, input_source, output_d, init_size=100, min_memory_radius=500, time_offsets=()):
        super().__init__(shape=output_d, time_offsets=time_offsets, min_memory_radius=min_memory_radius)
        self.output_d = output_d

        self.input_source: DataSource = input_source
        assert len(self.input_source.time_offsets) == 0
        self.pro = proSVD(k=output_d)


        l = []
        for _ in range(init_size):
            obs = next(self.input_source)
            l.append(obs)
        self.pro.initialize(np.array(l).T)


        for i in range(self.necessary_buffer[0]):
            obs = next(self.input_source)
            self.pro.preupdate()
            self.pro.updateSVD(obs[:,None])
            self.pro.postupdate()

            obs = obs @ self.pro.Q

            self.past.appendleft(obs)

        self.length = len(input_source) - init_size - sum(self.necessary_buffer)
        assert self.length >= 0

        self.simulate_more_steps()

    def __len__(self):
        return self.length


    def simulate_more_steps(self):
        # todo: might be better to do one at a time
        n = self.memory_radius - len(self.future)

        for _ in range(n):
            try:
                obs = next(self.input_source)
            except StopIteration:
                break
            self.pro.preupdate()
            self.pro.updateSVD(obs[:,None])
            self.pro.postupdate()

            obs = obs @ self.pro.Q
            self.future.append(obs)