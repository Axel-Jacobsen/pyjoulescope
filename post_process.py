import time
import math
import numpy as np
from typing import Callable
from collections import deque
from joulescope.data_recorder import DataReader

CHUNK_SIZE = 65536


class PostProcess:

    def __init__(self):
        self._c = None

    def histogram(self,
                  reader: DataReader,
                  t0: float = None,
                  t1: float = None,
                  num_bins: int = None,
                  signal: str = 'current',
                  norm: str = 'density') -> (np.array, float, float):
        """Creates a histogram of the data contained by the reader object, between t0 and t1

        :param reader: The opened DataReader object
        :param t0: start time of analysis, in seconds
        :param t1: end time of analysis, in seconds
        :param num_bins: number of bins for the histogram
        :param signal: the signal that will be analysied, either 'current' or 'voltage'
        :param norm: normalization of the histogram counts, either 'density' (integral over 
                     histogram is 1), 'unity' (sum over histograms is 1), or None (no normalization)

        :return: normalized array of values representing histogram (integral over the histogram is 1, 
                 gives probability density function) and an array of bin edges for the histogram
        """

        t0 = t0 if t0 else 0
        t1 = t1 if t1 else reader.duration

        id_start = reader.time_to_sample_id(t0)
        id_end = reader.time_to_sample_id(t1)

        statistics = reader.statistics_get(t0, t1)['signals'][signal]['statistics']
        maximum, minimum = statistics['max'], statistics['min']
        width = 3.5 * statistics['σ'] / (id_end - id_start)**(1. / 3)
        _num_bins = num_bins if num_bins is not None else math.ceil((maximum - minimum) / width)

        # if cache doesnt exist we cant use the cache
        # if num_bins is none, we use the cache (and our bins are the same as the caches bins)
        # if num_bins is not none, and if the number of bins is equal to the number of bins 
        #    of the cache, then we use the cache
        cache_exists = self._c is not None
        num_bins_doesnt_exist = num_bins is None
        num_bins_equal_cache_bins = num_bins is not None and (num_bins == len(self._c.bin_edges) - 1)

        if cache_exists and (num_bins_doesnt_exist or num_bins_equal_cache_bins):

            c_t0 = self._c.t0
            c_t1 = self._c.t1
            c_bin_edges = self._c.bin_edges

            # new hist is entirely before or after cached hist
            if (t0 < c_t0 and t1 < c_t0) or (c_t1 < t0 and c_t1 < t1):
                hist, bin_edges = self._calculate_histogram(reader, t0, t1, c_bin_edges, signal)
                if (t1 - t0) < self._c.duration:
                    self._c = HistogramCache(hist, t0, t1, c_bin_edges)
                return self._normalize_hist(hist, c_bin_edges, norm)
            
            # lower bound of new hist is below lower bound of cache, upper bound of new hist is in cache
            if t0 < c_t0 and c_t0 <= t1 <= c_t1:
                # if the upper bound is less than half way, its not worth it to use the cache
                if t1 < (c_t0 + c_t1) / 2:
                    lower_chunk, bin_edges = self._calculate_histogram(reader, t0, c_t0, c_bin_edges, signal)
                    upper_chunk, bin_edges = self._calculate_histogram(reader, c_t0, t1, c_bin_edges, signal)
                    hist_edges_out = self._normalize_hist(lower_chunk + upper_chunk, c_bin_edges, norm)
                else:
                    lower_chunk, bin_edges = self._calculate_histogram(reader, t0, c_t0, c_bin_edges, signal)
                    upper_chunk, bin_edges = self._calculate_histogram(reader, t1, c_t1, c_bin_edges, signal)
                    hist_edges_out = self._normalize_hist(lower_chunk + self._c.hist - upper_chunk, c_bin_edges, norm)
                self._c.hist += lower_chunk
                self._c.t0 = t0
                return hist_edges_out

            # all of new hist is in current cache
            if c_t0 <= t0 <= c_t1 and c_t0 <= t1 <= c_t1:
                # if the new hist less than what we would have to calculate with the cache, just calculate new hist
                if (t1 - t0) < self._c.duration / 2:
                    hist, bin_edges = self._calculate_histogram(reader, t0, t1, c_bin_edges, signal)
                    return self._normalize_hist(hist, c_bin_edges, norm)
                else:
                    lower_chunk, bin_edges = self._calculate_histogram(reader, c_t0, t0, c_bin_edges, signal)
                    upper_chunk, bin_edges = self._calculate_histogram(reader, t1, c_t1, c_bin_edges, signal)
                    return self._normalize_hist(self._c.hist - lower_chunk - upper_chunk, c_bin_edges, norm)

            # lower bound of new hist is in cache, upper is above it
            if c_t0 <= t0 <= c_t1 and c_t1 < t1:
                # if lower bound is in 
                if t0 < (c_t1 + c_t0) / 2:
                    lower_chunk, bin_edges = self._calculate_histogram(reader, c_t0, t0, c_bin_edges, signal)
                    upper_chunk, bin_edges = self._calculate_histogram(reader, c_t1, t1, c_bin_edges, signal)
                    hist_edges_out = self._normalize_hist(self._c.hist - lower_chunk + upper_chunk, c_bin_edges, norm)
                else:
                    lower_chunk, bin_edges = self._calculate_histogram(reader, t0, c_t1, c_bin_edges, signal)
                    upper_chunk, bin_edges = self._calculate_histogram(reader, c_t1, t1, c_bin_edges, signal)
                    hist_edges_out = self._normalize_hist(lower_chunk + upper_chunk, c_bin_edges, norm)
                self._c.hist += upper_chunk
                self._c.t1 = t1
                return hist_edges_out

        # If no cache could be used, then calculate the histogram without it
        hist, bin_edges = self._calculate_histogram(reader, t0, t1, _num_bins, signal)
        self._c = HistogramCache(hist, t0, t1, bin_edges)
        return self._normalize_hist(hist, bin_edges, norm)

    def _calculate_histogram(self,
                   reader: DataReader,
                   t0: float,
                   t1: float,
                   bins: np.array or int,
                   signal: str = 'current') -> (np.array, float, float):
        if t0 == t1:
            if isinstance(bins, np.ndarray):
                return np.zeros(len(bins) - 1), bins
            else:
                return np.array(), np.array()

        signal_index = _get_signal_index(signal)

        id_start = reader.time_to_sample_id(t0)
        id_end = reader.time_to_sample_id(t1)

        statistics = reader.statistics_get(t0, t1)['signals'][signal]['statistics']
        maximum, minimum = statistics['max'], statistics['min']

        _start = id_start
        _end = id_start + CHUNK_SIZE if id_start + CHUNK_SIZE < id_end else id_end

        data = reader.get_calibrated(_start, _end)
        hist, bin_edges = np.histogram(data[signal_index], range=(minimum, maximum), bins=bins)

        for i in range(_end, id_end, CHUNK_SIZE):
            print('progress: {:.3} %\t\r'.format(((i - id_start) / (id_end - id_start)) * 100), end='')

            start = i
            end = i + CHUNK_SIZE if i + CHUNK_SIZE < id_end else id_end

            data = reader.get_calibrated(start, end)
            hist += np.histogram(data[signal_index], range=(minimum, maximum), bins=bins)[0]

        return hist, bin_edges

    def max_window(self, reader: DataReader, duration: int, signal: str = 'current'):
        """Max Windowing function

        Finds the maximum current consumption over a certain duration for a dataset
        TODO add functionality for power

        :param reader: DataReader object with the data in question
        :param duration: lenth of window in seconds that will be analyzed
        :param signal: the signal that will be analysied, either 'current' or 'voltage'

        :returns: an array representing a cdf and its bin edges
        """
        signal_index = _get_signal_index(signal)
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
        """Cumulative Distribution function

        :param reader: DataReader object with the data in question
        :param signal: the signal that will be analysied, either 'current' or 'voltage'

        :returns: an array representing a cdf and its bin edges
        """
        hist, bin_edges = self.histogram(reader, signal=signal, norm='unity')
        _cdf = np.zeros(len(hist))
        for i, hist_val in enumerate(hist):
            _cdf[i] = _cdf[i - 1] + hist_val
        return _cdf, bin_edges

    def ccdf(self, reader: DataReader, signal: str = 'current'):
        """Complementary Cumulative Distribution Function

        :param reader: DataReader object with the data in question
        :param signal: the signal that will be analysied, either 'current' or 'voltage'

        :returns: an array representing a ccdf and its bin edges
        """
        _cdf, bin_edges = self.cdf(reader, signal=signal)
        return 1 - _cdf, bin_edges

    def _normalize_hist(self, hist, bin_edges, norm: str = 'density'):
        if norm == 'density':
            db = np.array(np.diff(bin_edges), float)
            return hist/db/hist.sum(), bin_edges
        elif norm == 'unity':
            return hist/hist.sum(), bin_edges
        elif norm == None:
            return hist, bin_edges
        else:
            raise RuntimeWarning('_normalize_hist invalid normalization; possible values are "density", "unity", or None')


class HistogramCache:
    def __init__(self, hist, t0, t1, bin_edges):
        self.hist = hist
        self.t0 = t0
        self.t1 = t1
        self.bin_edges = bin_edges

    @property
    def minimum(self):
        return self.bin_edges[0]

    @property
    def duration(self):
        return self.t1 - self.t0

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

    _c0 = 0
    _c1 = 10
    _t0 = 10
    _t1 = 3
    tf = reader.duration
    _c0s = [0,  10, 10,  0,  0,  0,  0,  0]
    _c1s = [1,  tf, tf, tf, tf, 20, 20, 10]
    _t0s = [10,  5,  5, 10,  5, 15,  5, 15]
    _t1s = [11, 15, 30, 20, 25, tf, tf, tf]

    i = 0
    for _c0, _c1, _t0, _t1 in zip(_c0s, _c1s, _t0s, _t1s):

        p = PostProcess()

        print(f'\nTEST NUMBER {i}')
        i+= 1

        t1 = time.time()
        v0s, bin_edges0 = p.histogram(reader, t0=_c0, t1=_c1)
        delt1 = time.time() - t1
        print('Full:', delt1)
        assert(np.sum(v0s*np.diff(bin_edges0)) == 1)

        t2 = time.time()
        v1s, bin_edges1 = p.histogram(reader, t0=_t0, t1=_t1)
        delt2 = time.time() - t2
        print('W/ Cache:', delt2)
        assert(np.array_equal(bin_edges0, bin_edges1))
        assert(abs(np.sum(v1s*np.diff(bin_edges1)) - 1) < 0.001)

        t3 = time.time()
        unnormalized, bin_edges2 = p._calculate_histogram(reader, _t0, _t1, bin_edges0)
        v2s, bin_edges2 = p._normalize_hist(unnormalized, bin_edges2)
        delt3 = time.time() - t3
        print('W/O Cache:', delt3)
        assert(np.array_equal(bin_edges1, bin_edges2))
        assert(abs(np.sum(v1s*np.diff(bin_edges1)) - 1) < 0.001)
        assert(np.array_equal(v1s, v2s))

        t4 = time.time()
        v4s, _ = p.histogram(reader, t0=_t0, t1=_t1, num_bins = 448, norm='unity')
        delt4 = time.time() - t4
        print('W/ unique bins:', delt4)
        assert(abs(v4s.sum() - 1) < 0.001)

def cdf_main():
    import matplotlib.pyplot as plt
    fname = '../sample_data/cleaner.jls'

    reader = DataReader()
    reader.open(fname)

    p = PostProcess()
    vs, bin_edges = p.cdf(reader)

    plt.plot(bin_edges[:-1], vs)
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
