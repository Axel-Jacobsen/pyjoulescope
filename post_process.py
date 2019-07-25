import time
import math
import numpy as np
from typing import Callable
from collections import deque
from joulescope.data_recorder import DataReader

CHUNK_SIZE = 2**16

class PostProcess:
    def __init__(self):
        self._hist_cache = None

    def histogram(self,
                  reader: DataReader,
                  t1: float = None,
                  t2: float = None,
                  signal: str = 'current') -> (np.array, float, float):
        """
        Creates a histogram of `signal` over time

        returns:
            hist: array of number of indcidences of data in that bin
            width: number to multiply bin index by to get the x value of the bin
            min: minimum data value
        """
        signal_index = _get_signal_index(signal)

        _t1 = t1 if t1 else 0
        _t2 = t2 if t2 else reader.duration

        id_start = reader.time_to_sample_id(_t1)
        id_end = reader.time_to_sample_id(_t2)

        statistics = reader.statistics_get( _t1, _t2)['signals'][signal]['statistics']
        maximum, minimum = statistics['max'], statistics['min']

        width = 3.5 * statistics['σ'] / (id_end - id_start)**(1. / 3)

        hist = histogram(width, maximum, minimum, signal, _t1, _t2)

        for i in range(id_start, id_end, CHUNK_SIZE):
            print('progress: {:.3} %\t\r'.format(((i - id_start) / (id_end - id_start)) * 100), end='')
            start = i
            end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end
            data = reader.get_calibrated(start, end)

            for v in data[signal_index]:
                hist.add_value(v)

        reader.close()

        return hist.get_histogram_array(), hist.width, hist.minimum

    def max_window(self, reader, duration: int):
        signal_index = _get_signal_index('current')
        id_start, id_end = reader.sample_id_range
        id_duration = reader.time_to_sample_id(duration)
        queue = deque(maxlen=id_duration)

        start = 0
        end = id_duration
        data = reader.get_calibrated(start, end)
        for i in range(id_start, id_duration):
            queue.append(data[signal_index][i])

        max = sum(queue)
        current_sum = max
        start_mark = 0
        end_mark = id_duration

        # There is probably some numpy way to do this that could also order you coffee at the same time,
        # but I am not sure what that could be and this is already fast enough for the point
        # of development that I am at currently
        for i in range(id_duration, id_end, CHUNK_SIZE):
            print('progress: {:.3} %\t\r'.format(((i) / id_end) * 100), end='')
            start = i
            end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end

            data = reader.get_calibrated(start, end)
            for j, new_value in enumerate(data[signal_index]):
                discard_val = queue.popleft()
                queue.append(new_value)
                current_sum += new_value - discard_val
                if current_sum > max:
                    max = current_sum
                    start_mark = i + j - id_duration
                    end_mark = i + j

        return max, start_mark, end_mark

    def cdf(self, reader: DataReader, signal: str = 'current'):
        """
        Cumulative Distribution function
        """
        hist, width, minimum = self.histogram(reader, signal=signal)

        _cdf = np.zeros(len(hist))
        _cdf[0] = hist[0]
        for i, hist_val in enumerate(hist, 1):
            _cdf[i] = _cdf[i - 1] + hist_val

        return _cdf, width, minimum

    def ccdf(self, reader: DataReader, signal: str = 'current'):
        """
        Complementary Cumulative Distribution Function
        """
        _cdf, width, minimum = self.cdf(reader, signal=signal)
        return 1 - _cdf, width, minimum


_signal_index = {'current': 0, 'voltage': 1}


def _get_signal_index(signal: str):
    if signal not in _signal_index.keys():
        raise RuntimeError(
            'Invalid Signal Request; possible values: "voltage", "current"')
    return _signal_index[signal]


def _get_max_min(reader_obj, id_start, id_end):
    maximum = 0
    minimum = 1e9  # assuming we are not consuming gigaamps or gigavolts through joulescope

    for i in range(id_start, id_end, CHUNK_SIZE):
        start = i
        end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end

        # data[0] is current, data[1] is voltage
        data = reader_obj.get_calibrated(start, end)
        for v in data[0]:
            maximum = max(v, maximum)
            minimum = min(v, minimum)

    return maximum, minimum


class histogram():

    def __init__(self, width, maximum, minimum, signal, t0=None, t1=None):
        self.num_bins = math.ceil((maximum - minimum) / width)
        self._hist = np.zeros(self.num_bins)
        self.width = width
        self.maximum = maximum
        self.minimum = minimum
        self.signal = signal
        self.t0 = t0
        self.t1 = t1
        self._num_entries = 0

    def _linear_bin(self, val):
        return math.floor((val - self.minimum) * (self.num_bins - 1) / (self.maximum - self.minimum))

    def add_value(self, val: float):
        self._num_entries += 1
        self._hist[self._linear_bin(val)] += 1

    def get_histogram_array(self):
        return self._hist / self._num_entries

def hist_main():
    import matplotlib.pyplot as plt
    fname = '../sample_data/cleaner.jls'

    reader = DataReader()
    reader.open(fname)

    p = PostProcess()
    vs, width, minimum = p.histogram(reader)
    xs = [i * width + minimum for i in range(len(vs))]

    plt.bar(xs, vs, width=width)
    plt.xlabel('Current (mA)')
    plt.ylabel('Probability')
    plt.show()

def cdf_main():
    import matplotlib.pyplot as plt
    fname = '../sample_data/cleaner.jls'

    reader = DataReader()
    reader.open(fname)

    p = PostProcess()
    vs, width, minimum = p.cdf(reader)
    xs = [i * width + minimum for i in range(len(vs))]

    plt.plot(xs, vs)
    plt.xlabel('Current (mA)')
    plt.ylabel('Probability')
    plt.show()


def window_main():
    fname = '../sample_data/cleaner.jls'

    reader = DataReader()
    reader.open(fname)

    p = PostProcess()
    t1 = time.time()
    max, start, end = p.max_window(reader, duration=1)
    print(time.time() - t1)
    print(max, start, end)


if __name__ == '__main__':
    hist_main()
    window_main()
