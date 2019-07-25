import time
import math
import numpy as np
from typing import Callable
from collections import deque
from joulescope.data_recorder import DataReader

CHUNK_SIZE = 65536

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
        bins = math.ceil((maximum - minimum) / width)

        _start = id_start
        _end = id_start + CHUNK_SIZE if id_start + CHUNK_SIZE < id_end else id_end

        data = reader.get_calibrated(_start, _end)
        hist, bin_edges = np.histogram(data[signal_index], range=(minimum,maximum), bins=bins)

        for i in range(_end, id_end, CHUNK_SIZE):
            print('progress: {:.3} %\t\r'.format(((i - id_start) / (id_end - id_start)) * 100), end='')

            start = i
            end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end

            data = reader.get_calibrated(start, end)
            hist += np.histogram(data[signal_index], range=(minimum, maximum), bins=bins)[0]

        reader.close()

        db = np.array(np.diff(bin_edges), float)

        return hist/db/hist.sum(), width, minimum

    def max_window(self, reader: DataReader, duration: int):
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
        for i, hist_val in enumerate(hist):
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


def hist_main():
    import matplotlib.pyplot as plt
    fname = '../sample_data/cleaner.jls'

    reader = DataReader()
    reader.open(fname)

    p = PostProcess()
    t1 = time.time()
    vs, width, minimum = p.histogram(reader)
    print(time.time() - t1)

    xs = [minimum + width * i  for i in range(len(vs))]

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
    cdf_main()
