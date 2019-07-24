import time
import math
import numpy as np
from joulescope.data_recorder import DataReader
from typing import Callable


class PostProcess:

    def __init__(self):
        pass

    def histogram(self,
                  fname: str,
                  t1: float = None,
                  t2: float = None,
                  signal: str = 'current') -> (np.array, float, float):
        """
        Creates a histogram of current over time
        Bin widths defined by Sturges' Formula - assuming an approximately
        normal distribution of data, and relatively simple compared to it's counterparts

        returns:
            hist: array of number of indcidences of data in that bin
            width: number to multiply bin index by to get the x value of the bin
            min: minimum data value
        """
        CHUNK_SIZE = 2**16
        signal_index = _get_signal_index(signal)

        if not fname.endswith('jls'):
            raise RuntimeError('file must be .jls type')

        reader = DataReader()
        reader.open(fname)

        _t1 = t1 if t1 else 0
        _t2 = t2 if t2 else reader.duration

        # start and end data id - i.e. range of data that we are calculating
        id_start = reader.time_to_sample_id(_t1)
        id_end = reader.time_to_sample_id(_t2)

        # Get statistics on the signal (i.e. voltage, current) that we want
        # `statistics` is a dictionary of μ, σ, max, min, p2p
        statistics = reader.statistics_get(_t1, _t2)['signals'][signal]['statistics']
        maximum, minimum = statistics['max'], statistics['min']  # when does max/min from statistics not equal max/min from _get_max_min?
        # print(minimum, maximum)
        # print(statistics['min'], statistics['max'])

        # num_bins = (1 + math.ceil(math.log(id_end, 2)))
        h = 3.5 * statistics['σ'] / (id_end - id_start)**(1. / 3)
        num_bins = math.ceil((maximum - minimum) / h)
        hist = np.zeros(num_bins)

        # _binning: real in [minimum, maximum] -> integer in [0, num_bins)
        _binning = lambda v: math.floor((v - minimum) * (num_bins - 1) / (maximum - minimum))

        for i in range(id_start, id_end, CHUNK_SIZE):
            print('progress: {:.3} %\t\r'.format((i / id_end) * 100), end='')
            start = i
            end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end

            data = reader.get_calibrated(start, end)

            for i in range(len(data[signal_index])):
                hist[_binning(data[signal_index][i])] += 1

        reader.close()
        hist /= (id_end - id_start)  # Normalize histogram bin counts
        width = h  #(maximum - minimum) / num_bins

        return hist, width, minimum

    def cdf(self, fname: str, signal: str = 'current'):
        """
        Cumulative Distribution function
        """
        hist, width, minimum = self.histogram(fname, signal)

        _cdf = np.zeros(len(hist))
        _cdf[0] = hist[0]
        for i in range(1, len(hist)):
            _cdf[i] = _cdf[i - 1] + hist[i]

        return _cdf, width, minimum

    def ccdf(self, fname: str, signal: str = 'current'):
        """
        Complementary Cumulative Distribution Function
        """
        _cdf, width, minimum = self.cdf(fname, signal)
        return 1 - _cdf, width, minimum


_signal_index = {'current': 0, 'voltage': 1}


def _get_signal_index(signal: str):
    if signal not in _signal_index.keys():
        raise RuntimeError(
            'Invalid Signal Request; possible values: "voltage", "current"')
    return _signal_index[signal]


def _get_max_min(reader_obj, id_start, id_end):
    CHUNK_SIZE = 2**16
    maximum = 0
    minimum = 1e9  # assuming we are not consuming gigaamps or gigavolts through joulescope

    for i in range(id_start, id_end, CHUNK_SIZE):
        start = i
        end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end

        # data[0] is current, data[1] is voltage
        data = reader_obj.get_calibrated(start, end)
        for i in range(len(data[0])):
            v = data[0][i]
            if v > maximum:
                maximum = v
            elif v < minimum:
                minimum = v

    return maximum, minimum


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    p = PostProcess()
    vs, width, minimum = p.ccdf('sample_data/cleaner.jls')
    xs = [i * width + minimum for i in range(len(vs))]

    plt.bar(xs, vs, width=width)
    plt.xlabel('Current (mA)')
    plt.ylabel('Probability')
    plt.show()
