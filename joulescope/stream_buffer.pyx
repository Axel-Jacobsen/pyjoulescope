# Copyright 2018 Jetperch LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optimized Cython native Joulescope code.
"""

# See https://cython.readthedocs.io/en/latest/index.html

# cython: boundscheck=False, wraparound=False, nonecheck=False, overflowcheck=False, cdivision=True

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t
from libc.float cimport FLT_MAX, FLT_MIN
from libc.math cimport isfinite, NAN

from libc.string cimport memset, memcpy
import logging
import numpy as np
cimport numpy as np


DEF PACKET_TOTAL_SIZE = 512
DEF PACKET_HEADER_SIZE = 8
DEF PACKET_PAYLOAD_SIZE = PACKET_TOTAL_SIZE - PACKET_HEADER_SIZE
DEF PACKET_INDEX_MASK = 0xffff
DEF PACKET_INDEX_WRAP = PACKET_INDEX_MASK + 1

DEF REDUCTION_MAX = 5
DEF SAMPLES_PER_PACKET = PACKET_PAYLOAD_SIZE // (2 * 2)

DEF RAW_SAMPLE_SZ = 2 * 2  # sizeof(uint16_t)
DEF CAL_SAMPLE_SZ = 2 * 4  # sizeof(float)

DEF STATS_FIELDS = 3  # current, voltage, power
DEF STATS_VALUES = 4  # mean, variance, min, max
DEF STATS_FLOATS_PER_SAMPLE = STATS_FIELDS * STATS_VALUES
DEF SUPPRESS_SAMPLES_DEFAULT = 3
DEF I_RANGE_D_LENGTH = 3


DEF SUPPRESS_MODE_OFF = 0  # disabled, zero delay
DEF SUPPRESS_MODE_NORMAL = 1


log = logging.getLogger(__name__)


NAME_TO_COLUMN = {
    'current': 0,
    'i': 0,
    'i_raw': 0,
    'voltage': 1,
    'v': 1,
    'v_raw': 1,
    'power': 2,
    'p': 2,
}


ctypedef void (*js_stream_buffer_cbk)(void * user_data, float * stats)


cdef struct js_stream_buffer_calibration_s:
    float current_offset[8]
    float current_gain[8]
    float voltage_offset[2]
    float voltage_gain[2]


cdef struct js_stream_buffer_reduction_s:
    int32_t enabled
    uint32_t samples_per_step
    uint32_t samples_per_reduction_sample
    uint32_t sample_counter
    uint32_t length
    js_stream_buffer_cbk cbk_fn
    void * cbk_user_data
    float *data    # data[length][3][4]  as [sample][i, v, power][mean, var, min, max]


cdef void stats_compute_reset(float stats[STATS_FIELDS][STATS_VALUES]):
    for i in range(STATS_FIELDS):
        stats[i][0] = 0.0  # mean
        stats[i][1] = 0.0  # variance
        stats[i][2] = FLT_MAX  # min
        stats[i][3] = -FLT_MAX  # max


cdef void stats_compute_one(float stats[STATS_FIELDS][STATS_VALUES],
                            float current,
                            float voltage):
    stats[0][0] += current
    if current < stats[0][2]:
        stats[0][2] = current
    if current > stats[0][3]:
        stats[0][3] = current

    stats[1][0] += voltage
    if voltage < stats[1][2]:
        stats[1][2] = voltage
    if voltage > stats[1][3]:
        stats[1][3] = voltage

    cdef float power = current * voltage
    stats[2][0] += power
    if power < stats[2][2]:
        stats[2][2] = power
    if power > stats[2][3]:
        stats[2][3] = power


cdef void stats_compute_end(float stats[STATS_FIELDS][STATS_VALUES],
                            float * data, uint32_t data_length,
                            uint64_t sample_id,
                            uint64_t length, uint64_t valid_length):
    cdef uint32_t k
    cdef uint32_t idx = sample_id % data_length
    if valid_length <= 0:
        return  # no samples so no update required!
    # compute mean
    cdef float scale = (<float> 1.0) / (<float> valid_length)
    stats[0][0] *= scale
    stats[1][0] *= scale
    stats[2][0] *= scale

    # compute variance
    cdef float i_mean = stats[0][0]
    cdef float v_mean = stats[1][0]
    cdef float p_mean = stats[2][0]
    cdef float i_var = 0.0
    cdef float v_var = 0.0
    cdef float p_var = 0.0
    cdef float t
    for count in range(length):
        k = 2 * idx
        if isfinite(data[k]):
            t = data[k] - i_mean
            i_var += t * t
            t = data[k + 1] - v_mean
            v_var += t * t
            t = data[k] * data[k + 1] - p_mean
            p_var += t * t
        idx += 1
        if idx >= data_length:
            idx = 0
    stats[0][1] = i_var * scale
    stats[1][1] = v_var * scale
    stats[2][1] = p_var * scale


cdef uint64_t stats_compute_run(
        float stats[STATS_FIELDS][STATS_VALUES],
        float * data, uint32_t data_length,
        uint64_t sample_id, uint64_t length):
    cdef uint32_t idx = sample_id % data_length
    cdef uint32_t data_idx
    cdef uint64_t counter = 0
    stats_compute_reset(stats)
    for i in range(length):
        data_idx = idx * 2
        if isfinite(data[data_idx]):
            stats_compute_one(stats, data[data_idx], data[data_idx + 1])
            counter += 1
        idx += 1
        if idx >= data_length:
            idx = 0
    stats_compute_end(stats, data, data_length, sample_id, length, counter)
    return counter


cdef uint64_t stats_combine(
        float stats[STATS_FIELDS][STATS_VALUES],
        uint64_t stats_sample_count,
        float stats_merge[STATS_FIELDS][STATS_VALUES],
        uint64_t stats_merge_sample_count):
    cdef uint64_t total_count = stats_sample_count + stats_merge_sample_count
    cdef double f1
    cdef double f2
    if 0 == total_count:
        stats_compute_reset(stats)
        return total_count
    for i in range(STATS_FIELDS):
        f1 = stats_sample_count / total_count
        f2 = 1.0 - f1
        mean_new = f1 * stats[i][0] + f2 * stats_merge[i][0]
        stats[i][1] = <float> (f1 * (stats[i][1] + (stats[i][0] - mean_new) ** 2) + \
            f2 * (stats_merge[i][1] + (stats_merge[i][0] - mean_new) ** 2))
        stats[i][0] = <float> mean_new
        stats[i][2] = min(stats[i][2], stats_merge[i][2])
        stats[i][3] = max(stats[i][3], stats_merge[i][3])
    return stats_sample_count + stats_merge_sample_count


cdef uint32_t reduction_stats(js_stream_buffer_reduction_s * r,
        float stats[STATS_FIELDS][STATS_VALUES], uint32_t idx_start, uint32_t length):
    cdef uint32_t count
    cdef uint32_t j
    cdef uint32_t valid = 0
    cdef uint32_t idx = idx_start
    cdef float * f
    cdef float scale
    cdef float i_mean
    cdef float v_mean
    cdef float p_mean
    cdef float i_var
    cdef float v_var
    cdef float p_var
    cdef float dv

    stats_compute_reset(stats)
    for count in range(length):
        f = r.data + idx * STATS_FLOATS_PER_SAMPLE
        if isfinite(f[0]):
            valid += 1
            for j in range(STATS_FIELDS):
                stats[j][0] += f[0]
                if f[2] < stats[j][2]:
                    stats[j][2] = f[2]
                if f[3] > stats[j][3]:
                    stats[j][3] = f[3]
                f += STATS_VALUES
        idx += 1
        if idx >= r.length:
            idx = 0

    if 0 == valid:
        for j in range(STATS_FIELDS):
            stats[j][0] = NAN
            stats[j][1] = NAN
            stats[j][2] = NAN
            stats[j][3] = NAN
    else:
        scale = (<float> 1.0) / (<float> valid)
        stats[0][0] *= scale
        stats[1][0] *= scale
        stats[2][0] *= scale

        idx = idx_start
        i_mean = stats[0][0]
        v_mean = stats[1][0]
        p_mean = stats[2][0]
        i_var = 0.0
        v_var = 0.0
        p_var = 0.0
        for count in range(length):
            f = r.data + idx * STATS_FLOATS_PER_SAMPLE
            if isfinite(f[0]):
                dv = f[0] - i_mean
                i_var += f[1] + dv * dv
                dv = f[4] - v_mean
                v_var += f[5] + dv * dv
                dv = f[8] - p_mean
                p_var += f[9] + dv * dv
            idx += 1
            if idx >= r.length:
                idx = 0
        stats[0][1] = i_var * scale
        stats[1][1] = v_var * scale
        stats[2][1] = p_var * scale
    return valid


cdef _reduction_downsample(js_stream_buffer_reduction_s * r,
        float * buffer, uint32_t idx_start, uint32_t idx_stop, uint32_t increment):
    cdef uint32_t idx = idx_start
    cdef float stats[STATS_FIELDS][STATS_VALUES]
    while idx + increment <= idx_stop:
        reduction_stats(r, stats, idx, increment)
        memcpy(buffer, stats, sizeof(stats))
        buffer += STATS_FLOATS_PER_SAMPLE
        idx += increment


def reduction_downsample(reduction, idx_start, idx_stop, increment):
    """Downsample a data reduction.

    :param reduction: The np.float32 (N, 3, 4) array.
    :param idx_start: The starting index (inclusive) in reduction.
    :param idx_stop: The stopping index (exclusive) in reduction.
    :param increment: The increment value
    :return: The downsampled reduction.

    The x-values can be constructed:

        x = np.arange(idx_start, idx_stop - increment + 1, increment, dtype=np.float64)
    """
    cdef js_stream_buffer_reduction_s r_inst
    r_inst.length = <uint32_t> len(reduction)
    length = (idx_stop - idx_start) // increment
    out = np.empty((length, 3, 4), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3, mode = 'c'] reduction_c = reduction
    r_inst.data = <float *> reduction_c.data

    out = np.empty((length, 3, 4), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3, mode = 'c'] out_c = out
    cdef float * out_ptr = <float *> out_c.data
    _reduction_downsample(&r_inst, out_ptr, idx_start, idx_stop, increment)
    return out



cdef class Statistics:

    cdef float stats[STATS_FIELDS][STATS_VALUES]
    cdef uint64_t length

    def __cinit__(self):
        stats_compute_reset(self.stats)
        self.length = 0

    def __init__(self, length=None, stats=None):
        if length is not None and stats is not None:
            self._init(stats)
            self.length = length

    def __len__(self):
        return self.length

    cdef _init(self, stats):
        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] stats_c = stats
        cdef float * stats_ptr = <float *> stats_c.data
        memcpy(self.stats, stats_ptr, sizeof(self.stats))

    def combine(self, other: Statistics):
        self.length = stats_combine(self.stats, self.length, other.stats, other.length)
        return self

    cdef _value(self):
        out = np.empty((3, 4), dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] out_c = out
        cdef float * out_ptr = <float *> out_c.data
        memcpy(out_ptr, self.stats, sizeof(self.stats))
        return out

    @property
    def value(self):
        return self._value()


cdef void cal_init(js_stream_buffer_calibration_s * self):
    for i in range(8):
        self.current_offset[i] = <float> 0.0
        self.current_gain[i] = <float> 1.0
    self.current_gain[7] = 0.0  # always compute zero current when off
    for i in range(2):
        self.voltage_offset[i] = <float> 0.0
        self.voltage_gain[i] = <float> 1.0


cdef class StreamBuffer:
    """Efficient real-time Joulescope data buffering.

    :param length: The total length of the buffering in samples.
    :param reductions: The list of reduction integers.  Each integer represents
        the reduction amount for each resuting sample in units of samples
        of the previous reduction.  Reduction 0 is in raw sample units.
    """

    cdef uint32_t reduction_step
    cdef uint32_t length # in samples
    cdef uint64_t packet_index
    cdef uint64_t packet_index_offset
    cdef uint64_t device_sample_id
    cdef uint64_t processed_sample_id
    cdef uint64_t sample_missing_count  # based upon sample_id
    cdef uint64_t skip_count            # number of sample skips
    cdef uint64_t sample_sync_count     # based upon alternating 0/1 pattern
    cdef uint64_t contiguous_count      #
    cdef uint16_t *raw_ptr  # raw[length][2]   as i, v
    cdef float *data_ptr    # data[length][2]  as i, v
    cdef js_stream_buffer_calibration_s cal
    cdef js_stream_buffer_reduction_s reductions[REDUCTION_MAX]
    cdef uint32_t reduction_count

    cdef uint8_t i_range_d[I_RANGE_D_LENGTH]  # the i_range delay for processing
    cdef int32_t suppress_samples  # the total number of samples to suppress after range change
    cdef int32_t suppress_count  # the suppress counter, 1 = replace previous
    cdef uint8_t _suppress_mode

    cdef uint32_t stats_counter  # excludes NAN for mean
    cdef uint32_t stats_remaining
    cdef float stats[STATS_FIELDS][STATS_VALUES]  # [i, v, power][mean, var, min, max]

    cdef uint16_t sample_toggle_last
    cdef uint16_t sample_toggle_mask
    cdef uint8_t voltage_range

    cdef object raw
    cdef object data
    cdef object reductions_data
    cdef uint64_t _sample_id_max  # used to automatically stop streaming
    cdef uint64_t _contiguous_max  # used to automatically stop streaming
    cdef object _callback  # fn(np.array [3][4] of statistics, energy)
    cdef object _charge_picocoulomb  # python integer for infinite precision
    cdef object _energy_picojoules  # python integer for infinite precision


    def __cinit__(self, length, reductions):
        cdef uint32_t r_samples = 1
        if length < SAMPLES_PER_PACKET:
            raise ValueError('length to small')
        if len(reductions) > REDUCTION_MAX:
            raise ValueError('too many reductions')

        self._charge_picocoulomb = 0
        self._energy_picojoules = 0
        memset(self.reductions, 0, sizeof(self.reductions))

        # round up length to multiple of reductions
        self.reduction_step = int(np.prod(reductions))
        length = int(np.ceil(length / self.reduction_step)) * self.reduction_step
        self.length = length
        cal_init(&self.cal)

        self.raw = np.empty((length * 2), dtype=np.uint16)
        self.raw = np.ascontiguousarray(self.raw, dtype=np.uint16)
        cdef np.ndarray[np.uint16_t, ndim=1, mode = 'c'] raw_c = self.raw
        self.raw_ptr = <uint16_t *> raw_c.data

        self.data = np.empty((length * 2), dtype=np.float32)
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] data_c = self.data
        self.data_ptr = <float *> data_c.data

        self.reductions_data = []
        cdef js_stream_buffer_reduction_s * r
        cdef np.ndarray[np.float32_t, ndim=3, mode = 'c'] reduction_data_c
        sz = length


        for idx, rsamples in enumerate(reductions):
            sz = sz // rsamples
            r = &self.reductions[idx]
            r.enabled = 1
            r.samples_per_step = <uint32_t> rsamples
            r.length = sz
            r_samples *= rsamples
            r.samples_per_reduction_sample = r_samples

            d = np.empty((sz, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
            d = np.ascontiguousarray(d, dtype=np.float32)
            reduction_data_c = d
            r.data = <float *> reduction_data_c.data
            self.reductions_data.append(d)

        self.reduction_count = <uint32_t> len(reductions)
        if len(reductions):
            self.reductions[len(reductions) - 1].cbk_fn = _on_cbk
            self.reductions[len(reductions) - 1].cbk_user_data = <void *> self

        self._sample_id_max = 0  # used to automatically stop streaming
        self._contiguous_max = 0  # used to automatically stop streaming
        self._callback = None  # fn(np.array [3][4] of statistics, energy)
        self._charge_picocoulomb = 0
        self._energy_picojoules = 0  # integer for infinite precision

        self.suppress_samples = SUPPRESS_SAMPLES_DEFAULT
        self.suppress_count = 0
        self._suppress_mode = SUPPRESS_MODE_NORMAL

        for idx in range(I_RANGE_D_LENGTH):
            self.i_range_d[idx] = 7

    def __init__(self, length, reductions):
        self.reset()

    def __len__(self):
        return self.length

    def __str__(self):
        reductions = []
        for idx in range(REDUCTION_MAX):
            if self.reductions[idx].enabled:
                reductions.append(self.reductions.samples_per_step)
        return 'StreamBuffer(length=%d, reductions=%r)' % (self.length, reductions)

    @property
    def sample_id_range(self):
        """Get the range of sample ids currently available in the buffer.

        :return: Tuple of sample_id start, sample_id end.
        """
        s_end = int(self.processed_sample_id)
        s_start = s_end - self.length
        if s_start < 0:
            s_start = 0
        return s_start, s_end

    @property
    def data_buffer(self):
        return self.data  # the cdef np.ndarray

    @property
    def sample_id_max(self):
        return self._sample_id_max

    @sample_id_max.setter
    def sample_id_max(self, value):
        self._sample_id_max = value

    @property
    def contiguous_max(self):
        return self._contiguous_max

    @contiguous_max.setter
    def contiguous_max(self, value):
        self._contiguous_max = value

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value

    @property
    def voltage_range(self):
        return self.voltage_range

    @property
    def suppress_mode(self):
        if self._suppress_mode == SUPPRESS_MODE_OFF:
            return 'off'
        elif self._suppress_mode == SUPPRESS_MODE_NORMAL:
            return 'normal'
        else:
            return 'off'

    @suppress_mode.setter
    def suppress_mode(self, value):
        if isinstance(value, str):
            value = value.lower()
        if value in [SUPPRESS_MODE_OFF, 'off', False] or value is None:
            self._suppress_mode = SUPPRESS_MODE_OFF
        elif value in [SUPPRESS_MODE_NORMAL, 'normal', 'on', True]:
            self._suppress_mode = SUPPRESS_MODE_NORMAL
        else:
            raise ValueError('unsupported mode: %s' % (value, ))

    def status(self):
        return {
            'device_sample_id': {'value': self.device_sample_id, 'units': 'samples'},
            'sample_id': {'value': self.processed_sample_id, 'units': 'samples'},
            'sample_missing_count': {'value': self.sample_missing_count, 'units': 'samples'},
            'skip_count': {'value': self.skip_count, 'units': ''},
            'sample_sync_count': {'value': self.sample_sync_count, 'units': 'samples'},
            'contiguous_count': {'value': self.contiguous_count, 'units': 'samples'},
        }

    def calibration_set(self, current_offset, current_gain, voltage_offset, voltage_gain):
        cdef js_stream_buffer_calibration_s * cal = &self.cal
        for i in range(7):
            cal.current_offset[i] = current_offset[i]
            cal.current_gain[i] = current_gain[i]
        cal.current_offset[7] = 0.0
        cal.current_gain[7] = 0.0
        for i in range(2):
            cal.voltage_offset[i] = voltage_offset[i]
            cal.voltage_gain[i] = voltage_gain[i]

    cdef _stats_reset(self):
        self.stats_counter = 0
        self.stats_remaining = 0
        if self.reductions[0].enabled:
            self.stats_remaining = self.reductions[0].samples_per_step
        stats_compute_reset(self.stats)

    def reset(self):
        self.packet_index = 0
        self.packet_index_offset = 0
        self.device_sample_id = 0
        self.processed_sample_id = 0
        self.sample_missing_count = 0
        self.skip_count = 0
        self.sample_sync_count = 0
        self.contiguous_count = 0
        self.sample_toggle_last = 0
        self.sample_toggle_mask = 0
        self.voltage_range = 0
        self._sample_id_max = 1 << 63  # big enough
        self._contiguous_max = 1 << 63  # big enough
        self._charge_picocoulomb = 0
        self._energy_picojoules = 0
        self.stats_counter = 0
        self.suppress_count = 0
        for idx in range(I_RANGE_D_LENGTH):
            self.i_range_d[idx] = 7
        for idx in range(REDUCTION_MAX):
            self.reductions[idx].sample_counter = 0
        self._stats_reset()

    cdef uint32_t reduction_index(self, js_stream_buffer_reduction_s * r, uint32_t parent_samples_per_step):
        cdef uint32_t idx = <uint32_t> (self.processed_sample_id % self.length)
        cdef uint32_t samples_per_step = parent_samples_per_step * r.samples_per_step;
        idx /= samples_per_step
        if 0 == idx:
            idx = r.length - 1
        else:
            idx -= 1
        return idx

    cdef void reduction_update_n(self, int n, uint32_t parent_samples_per_step):
        cdef js_stream_buffer_reduction_s * r = &self.reductions[n]
        cdef float stats[STATS_FIELDS][STATS_VALUES]
        cdef uint32_t samples_per_step
        cdef uint32_t idx_target
        cdef uint32_t idx_start
        cdef uint32_t data_idx

        if 0 == r.enabled:
            return
        r.sample_counter += 1
        if r.sample_counter >= r.samples_per_step:
            r.sample_counter = 0
            samples_per_step = parent_samples_per_step * r.samples_per_step
            idx_target = self.reduction_index(r, parent_samples_per_step)
            idx_start = idx_target * r.samples_per_step
            data_idx = idx_target * <uint32_t> (sizeof(stats) / sizeof(float))
            reduction_stats(&self.reductions[n - 1], stats, idx_start, r.samples_per_step)
            memcpy(r.data + data_idx, stats, sizeof(stats))
            if r.cbk_fn:
                r.cbk_fn(r.cbk_user_data, r.data + data_idx)
            self.reduction_update_n(n + 1, samples_per_step)

    cdef void reduction_update_0(self):
        cdef js_stream_buffer_reduction_s * r = &self.reductions[0]
        if 0 == r.enabled:
            return
        cdef uint32_t idx = self.reduction_index(r, 1)
        memcpy(r.data + idx * sizeof(self.stats) // sizeof(float), self.stats, sizeof(self.stats))
        # note: samples_per_step is not statistically correct on missing samples.
        self.reduction_update_n(1, self.reductions[0].samples_per_step)

    cdef void stats_finalize(self):
        if 0 == self.stats_counter:
            for i in range(STATS_FIELDS):
                self.stats[i][0] = NAN
                self.stats[i][1] = NAN
                self.stats[i][2] = NAN
                self.stats[i][3] = NAN
            return
        cdef uint32_t length = self.reductions[0].samples_per_step
        cdef uint32_t idx = self.reduction_index(&self.reductions[0], 1)
        idx *= self.reductions[0].samples_per_step
        stats_compute_end(self.stats, self.data_ptr, self.length, idx, length, self.stats_counter)

    cdef void _insert_usb_bulk(self, const uint8_t *data, size_t length):
        cdef uint8_t buffer_type
        cdef uint8_t status
        cdef uint16_t pkt_length
        cdef uint64_t pkt_index
        cdef uint64_t sample_id
        cdef uint32_t idx
        cdef uint32_t idx2
        cdef uint32_t samples
        cdef uint32_t samples_to_end

        while length >= PACKET_TOTAL_SIZE:
            buffer_type = data[0]
            status = data[1]
            pkt_length = (data[2] | ((<uint16_t> data[3] & 0x7f) << 8)) & 0x7fff
            self.voltage_range = <uint8_t> ((data[3] >> 7) & 0x01)
            pkt_index = <uint64_t> (data[4] | ((<uint16_t> data[5]) << 8))
            # uint16_t usb_frame_index = data[6] | ((<uint16_t> data[7]) << 8)
            length -= PACKET_TOTAL_SIZE

            if (1 != buffer_type) or (0 != status) or (PACKET_TOTAL_SIZE != pkt_length):
                data += PACKET_TOTAL_SIZE
                continue
            pkt_index += self.packet_index_offset
            while pkt_index < self.packet_index:
                pkt_index += PACKET_INDEX_WRAP
                self.packet_index_offset += PACKET_INDEX_WRAP
            sample_id = pkt_index * SAMPLES_PER_PACKET
            idx = self.device_sample_id % self.length

            if sample_id < self.device_sample_id:
                log.warning("WARNING: duplicate data")
            elif self.device_sample_id < sample_id:
                log.info("Fill missing samples: %r, %r", self.device_sample_id, sample_id)
                self.skip_count += 1
                self.contiguous_count = 0
                while self.device_sample_id < sample_id:
                    idx2 = idx * 2
                    self.raw[idx2 + 0] = 0xffff  # missing sample is i_range 7 with raw_i = 0x3fff
                    self.raw[idx2 + 1] = 0xffff
                    idx += 1
                    if idx >= self.length:
                        idx = 0
                    self.device_sample_id += 1
                    self.sample_missing_count += 1

            samples = SAMPLES_PER_PACKET
            self.contiguous_count += samples
            data += PACKET_HEADER_SIZE  # skip header
            if (idx + SAMPLES_PER_PACKET) > self.length:
                samples_to_end = self.length - idx
                memcpy(&self.raw_ptr[idx * 2], data, samples_to_end * RAW_SAMPLE_SZ)
                idx = 0
                data += samples_to_end * RAW_SAMPLE_SZ
                samples -= samples_to_end
            memcpy(&self.raw_ptr[idx * 2], data, samples * RAW_SAMPLE_SZ)
            data += samples * RAW_SAMPLE_SZ
            self.device_sample_id = sample_id + SAMPLES_PER_PACKET
            self.packet_index += 1

    cdef _check_stop(self):
        cdef bint duration_stop = self.device_sample_id >= self._sample_id_max
        cdef bint contiguous_stop = self.contiguous_count >= self._contiguous_max
        rv = duration_stop or contiguous_stop
        if rv:
            if duration_stop:
                log.info('insert causing duration stop %d >= %d',
                         self.device_sample_id, self._sample_id_max)
            elif duration_stop:
                log.info('insert causing contiguous stop %d >= %d',
                         self.contiguous_count, self._contiguous_max)
        return rv

    cpdef insert(self, data):
        """Insert new device USB data into the buffer.

        :param data: The new data to insert.
        :return: False to continue streaming, True to end.
        """
        cdef np.ndarray[np.uint8_t, ndim=1, mode = 'c'] data_c
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data, dtype=np.uint8)
            data_c = data
            data_ptr = <uint8_t *> data_c.data
            self._insert_usb_bulk(data_ptr, len(data))
        else:
            self._insert_usb_bulk(data, len(data))
        return self._check_stop()

    cpdef insert_raw(self, data):
        """Insert raw data into the buffer
        
        :param data: The np.array of np.uint16 data to insert.
        :return: False to continue streaming, True to end.
        """
        if data.dtype != np.uint16:
            raise ValueError('raw data must np np.uint16 array')
        data = data.reshape((-1, ))
        sample_count = len(data)
        if sample_count % 2:
            raise ValueError('raw data must be multiples of 2 16-bit values')
        sample_count = sample_count // 2
        log.info('insert_raw %d', sample_count)
        idx = self.device_sample_id % self.length
        sample_count_remaining = sample_count
        while idx + sample_count_remaining > self.length:
            samples_to_end = self.length - idx
            self.raw[idx * 2:] = data[:samples_to_end * 2]
            data = data[samples_to_end * 2:]
            idx = 0
            sample_count_remaining -= samples_to_end
        if sample_count_remaining:
            self.raw[idx * 2: (idx + sample_count_remaining) * 2] = data
        self.contiguous_count += sample_count
        self.device_sample_id += sample_count
        return self._check_stop()

    cdef void _process(self):
        cdef uint32_t idx_start
        cdef uint32_t idx
        cdef uint32_t suppress_idx
        cdef uint32_t i_range_idx
        cdef uint16_t raw_i
        cdef uint16_t raw_v
        cdef uint8_t i_range
        cdef uint16_t sample_toggle_current
        cdef uint64_t sample_sync_count
        cdef float cal_i
        cdef float cal_v

        if self.processed_sample_id + self.length < self.device_sample_id:
            log.warning('process: stream_buffer is behind: %r + %r < %r',
                        self.processed_sample_id, self.length, self.device_sample_id)
            self.processed_sample_id = self.device_sample_id - self.length
        idx_start = <uint32_t> (self.processed_sample_id % self.length)

        while self.processed_sample_id < self.device_sample_id:
            idx = idx_start * 2
            raw_i = self.raw_ptr[idx + 0]
            raw_v = self.raw_ptr[idx + 1]
            for i_range_idx in range(I_RANGE_D_LENGTH - 1, 0, -1):
                self.i_range_d[i_range_idx] = self.i_range_d[i_range_idx - 1]
            self.i_range_d[0] = <uint8_t> ((raw_i & 0x0003) | ((raw_v & 0x0001) << 2))

            # process i_range for glitch suppression
            if SUPPRESS_MODE_NORMAL == self._suppress_mode:
                if self.i_range_d[0] == self.i_range_d[2]:
                    # no change, use single delay since i_range leads data by 1 sample
                    i_range = self.i_range_d[1]
                elif self.i_range_d[0] == 0x7:
                    # select is off or missing sample, use immediately
                    i_range = self.i_range_d[0]
                elif self.i_range_d[0] < self.i_range_d[2]:
                    # use old select one more sample
                    # moving to less sensitive range (smaller value resistor)
                    i_range = self.i_range_d[2]
                    if self.i_range_d[1] < self.i_range_d[2]:
                        # delay suppress by one sample
                        self.suppress_count = self.suppress_samples + 1
                else:
                    # Use new select immediately
                    # moving to more sensitive range (larger value resistor)
                    i_range = self.i_range_d[0]
                    self.suppress_count = self.suppress_samples + 1
            else:
                i_range = self.i_range_d[0]

            if not 0 <= i_range <= 7:  # should never happen
                log.warning('i_range out of range: %s', i_range)
                i_range = 7

            sample_toggle_current = (raw_v >> 1) & 0x1
            raw_i = raw_i >> 2
            raw_v = raw_v >> 2
            sample_sync_count = (sample_toggle_current ^ self.sample_toggle_last ^ 1) & \
                    self.sample_toggle_mask
            if sample_sync_count:
                self.contiguous_count = 0
            self.sample_sync_count += sample_sync_count
            self.sample_toggle_last = sample_toggle_current
            self.sample_toggle_mask = 0x1
            cal_i = <float> raw_i
            cal_i += self.cal.current_offset[i_range]
            cal_i *= self.cal.current_gain[i_range]

            cal_v = <float> raw_v
            cal_v += self.cal.voltage_offset[self.voltage_range]
            cal_v *= self.cal.voltage_gain[self.voltage_range]
            if 7 == i_range:
                if 0x3fff == raw_i and 0x3fff == raw_v:  # missing sample
                    cal_i = NAN
                    cal_v = NAN
                else:  # select off, current is zero by definition
                    cal_i = <float> 0.0
            self.data_ptr[idx + 0] = cal_i
            self.data_ptr[idx + 1] = cal_v

            # Suppress Joulescope range switching glitch (at least for now).
            self.processed_sample_id += 1

            if self.suppress_count > 0:  # disable
                if self.suppress_count == 1:
                    suppress_idx = idx_start + self.length - self.suppress_samples
                    while suppress_idx >= self.length:
                        suppress_idx -= self.length
                    for a in range(self.suppress_samples + 1):
                        self.data_ptr[2 * suppress_idx + 0] = cal_i
                        self.data_ptr[2 * suppress_idx + 1] = cal_v
                        suppress_idx += 1
                        if suppress_idx >= self.length:
                            suppress_idx = 0
                        self._process_stats(cal_i, cal_v)
                else:
                    # temporarily write NaN, reprocess at end, above
                    self.data_ptr[idx + 0] = NAN
                    self.data_ptr[idx + 1] = NAN
                self.suppress_count -= 1
            else:
                self._process_stats(cal_i, cal_v)

            idx_start += 1
            if idx_start >= self.length:
                idx_start = 0

    cdef void _process_stats(self, cal_i, cal_v):
        if 0 == self.reductions[0].enabled:
            return
        if isfinite(cal_i):
            self.stats_counter += 1
            stats_compute_one(self.stats, cal_i, cal_v)
        if self.stats_remaining > self.reductions[0].samples_per_step:
            log.warning('Internal error stats_remaining: %d > %d',
                        self.stats_remaining,
                        self.reductions[0].samples_per_step)
            self._stats_reset()
        elif self.stats_remaining > 1:
            self.stats_remaining -= 1
        elif self.stats_remaining <= 1:
            self.stats_finalize()
            self.reduction_update_0()
            self._stats_reset()

    def process(self):
        self._process()

    cdef int range_check(self, uint64_t start, uint64_t stop):
        if stop <= start:
            log.warning("js_stream_buffer_get stop <= start")
            return 0
        if start > self.processed_sample_id:
            log.warning("js_stream_buffer_get start newer that current")
            return 0
        if stop > self.processed_sample_id:
            log.warning("js_stream_buffer_get stop newer than current")
            return 0
        return 1

    def stats_get(self, start, stop, out=None):
        """Get exact statistics over the specified range.

        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :param out: The optional output array.  None (default) creates
            and outputs a new array.
        :return: The np.ndarray((3, 4), dtype=np.float32) data of
            (fields, values) with
            fields (current, voltage, power) and
            values (mean, variance, min, max).
        """
        cdef uint64_t length
        cdef uint64_t count
        cdef float stats_accum[STATS_FIELDS][STATS_VALUES]
        cdef float stats_merge[STATS_FIELDS][STATS_VALUES]
        cdef uint32_t samples_per_step

        # log.info('stats_get(%r, %r)', start, stop)
        if start < 0 or stop < 0:
            log.warning('sample_id < 0: %d, %d', start, stop)
            return None
        if not self.range_check(start, stop):
            return None
        if stop <= start:
            log.warning('stop <= start: %d <= %d', start, stop)
            return None

        stats_compute_reset(stats_accum)
        length = stop - start
        if out is None:
            out = np.empty((STATS_FIELDS, STATS_VALUES), dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] out_c = out

        ranges = [[start, stop], [None, None]]

        sample_count = 0
        for n in range(self.reduction_count - 1, -1, -1):
            samples_per_step = self.reductions[n].samples_per_reduction_sample
            for idx, [r1, r2] in enumerate(ranges):
                if r1 is None:
                    continue
                k1 = r1 // samples_per_step * samples_per_step
                if k1 < r1:
                    k1 += samples_per_step
                k2 = (r2 // samples_per_step) * samples_per_step
                if k1 < k2:  # we can use this reduction!
                    # log.info('reduction %d on %d: %s to %s', n, idx, k1, k2)
                    r_idx_start = <uint32_t> ((k1 % self.length) // samples_per_step)
                    r_sample_length = k2 - k1
                    r_idx_length = r_sample_length // samples_per_step
                    reduction_stats(&self.reductions[n], stats_merge, r_idx_start, r_idx_length)
                    sample_count = stats_combine(stats_accum, sample_count, stats_merge, r_sample_length)
                    if idx == 0:
                        if r1 == k1:
                            ranges[idx] = [None, None]
                        else:
                            ranges[idx] = [r1, k1]
                        if ranges[1][0] is None and k2 < r2:
                            ranges[1] = [k2, r2]
                    else:
                        if r2 == k2:
                            ranges[idx] = [None, None]
                        else:
                            ranges[idx] = [k2, r2]

        # log.info('ranges = %r', ranges)
        for r1, r2 in ranges:
            if r1 is not None:
                r_sample_length = r2 - r1
                count = stats_compute_run(stats_merge, self.data_ptr, self.length, r1, r_sample_length)
                # log.info('direct: %s to %s (%d + %d)', r1, r2, sample_count, count)
                sample_count = stats_combine(stats_accum, sample_count, stats_merge, count)

        memcpy(out_c.data, stats_accum, sizeof(stats_accum))
        return out

    cdef uint64_t _data_get(self, float * buffer, uint64_t buffer_samples,
                            int64_t start, int64_t stop, uint64_t increment):
        """Get the summarized statistics over a range.
        
        :param buffer: The Nx3x4 buffer to populate.
        :param buffer_samples: The size N of the buffer in units of 3x4 float samples.
        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param increment: The number of raw samples.
        :return: The number of samples placed into buffer.
        """
        cdef uint64_t buffer_samples_orig = buffer_samples
        cdef int64_t idx
        cdef int64_t data_offset
        cdef float stats[STATS_FIELDS][STATS_VALUES]
        cdef uint64_t count
        cdef uint64_t fill_count = 0
        cdef uint64_t fill_count_tmp
        cdef uint64_t samples_per_step
        cdef uint64_t samples_per_step_next
        cdef uint64_t length
        cdef int64_t idx_start
        cdef int64_t end_gap
        cdef int64_t start_orig = start
        cdef uint64_t n
        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] out_c

        if stop + self.length < (<int64_t> self.processed_sample_id):
            fill_count = buffer_samples_orig
        elif start < 0:
            # round to floor, absolute value
            fill_count_tmp = ((-start + increment - 1) // increment)
            start += fill_count_tmp * increment
            #log.info('_data_get start < 0: %d [%d] => %d', start_orig, fill_count_tmp, start)
            fill_count += fill_count_tmp

        if not self.range_check(start, stop):
            return 0

        if (start + self.length) < (<int64_t> self.device_sample_id):
            fill_count_tmp = (self.device_sample_id - (start + self.length)) // increment
            start += fill_count_tmp * increment
            #log.info('_data_get behind < 0: %d [%d] => %d', start_orig, fill_count_tmp, start)
            fill_count += fill_count_tmp

        # Fill in too old of data with NAN
        for n in range(fill_count):
            if buffer_samples == 0:
                log.warning('_data_get filled with NaN %d of %d', buffer_samples_orig, fill_count)
                return buffer_samples_orig
            for j in range(STATS_FLOATS_PER_SAMPLE):
                buffer[j] = NAN
            buffer += STATS_FLOATS_PER_SAMPLE
            buffer_samples -= 1
        if buffer_samples != buffer_samples_orig:
            log.warning('_data_get filled %s', buffer_samples_orig - buffer_samples)

        if increment <= 1:
            # direct copy
            idx = start % self.length
            while start != stop and buffer_samples:
                data_offset = idx * 2
                buffer[0] = self.data_ptr[data_offset]
                buffer[1] = <float> 0.0
                buffer[2] = NAN
                buffer[3] = NAN
                buffer[4] = self.data_ptr[data_offset + 1]
                buffer[5] = <float> 0.0
                buffer[6] = NAN
                buffer[7] = NAN
                buffer[8] = self.data_ptr[data_offset] * self.data_ptr[data_offset + 1]
                buffer[9] = <float> 0.0
                buffer[10] = NAN
                buffer[11] = NAN
                buffer_samples -= 1
                idx += 1
                start += 1
                buffer += STATS_FLOATS_PER_SAMPLE
                if idx >= self.length:
                    idx = 0
        elif not self.reductions[0].enabled or (self.reductions[0].samples_per_step > increment):
            # compute over raw data.
            while start + <int64_t> increment <= stop and buffer_samples:
                count = stats_compute_run(stats, self.data_ptr, self.length, start, increment)
                memcpy(buffer, stats, sizeof(stats))
                buffer += STATS_FLOATS_PER_SAMPLE
                start += increment
                buffer_samples -= 1
        else:
            # use reductions through stats_get
            out = np.empty((STATS_FIELDS, STATS_VALUES), dtype=np.float32)
            out_c = out
            while start + <int64_t> increment <= stop and buffer_samples:
                next_start = start + increment
                self.stats_get(start, next_start, out)
                memcpy(buffer, out_c.data, STATS_FLOATS_PER_SAMPLE * sizeof(float))
                buffer += STATS_FLOATS_PER_SAMPLE
                start += increment
                buffer_samples -= 1
        return buffer_samples_orig - buffer_samples

    def data_get(self, start, stop, increment=None, out=None):
        """Get the samples with statistics.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param increment: The number of raw samples.
        :param out: The optional output np.ndarray(N, 3, 4) to populate with
            the result.  None (default) will construct and return a new array.
        :return: The np.ndarray((N, 3, 4), dtype=np.float32) data of
            (length, fields, values) with
            fields (current, voltage, power) and
            values (mean, variance, min, max).
        """
        increment = 1 if increment is None else int(increment)
        if start >= stop:
            log.info('data_get: start >= stop')
            return np.empty((0, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
        expected_length = (stop - start) // increment
        if out is None:
            out = np.empty((expected_length, STATS_FIELDS, STATS_VALUES), dtype=np.float32)

        #out = np.ascontiguousarray(out, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=3, mode = 'c'] out_c = out
        out_ptr = <float *> out_c.data

        length = self._data_get(out_ptr, len(out), start, stop, increment)
        if length != expected_length:
            log.warning('length mismatch: expected=%s, returned=%s', expected_length, length)
        return out[:length, :, :]

    def raw_get(self, start, stop):
        """Get the raw data from Joulescope.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :return: The np.ndarray((2 * N), dtype=np.uint16) data of
            interleaved current, voltage left-shifted 14-bit samples.
            The least significant 2 bits contain current range select
            information.
        """
        if stop <= start:
            log.warning('raw %d <= %d', start, stop)
            return np.empty((0, 2), dtype=np.uint16)
        total_length = self.length
        start_idx = start % total_length
        stop_idx = stop % total_length
        if 0 == stop_idx:
            stop_idx = total_length
        if stop_idx > start_idx:
            return self.raw[(start_idx * 2):(stop_idx * 2)].reshape((-1, 2))
        # on wrap, have to copy
        expected_length = stop - start
        samples1 = self.length - start_idx
        samples2 = expected_length - samples1
        out = np.empty((expected_length, 2), dtype=np.uint16)
        raw = self.raw.reshape((-1, 2))
        out[:samples1, :] = raw[start_idx:, :]
        out[samples1:, :] = raw[:samples2, :]
        return out

    def get_reduction(self, idx, start, stop):
        """Get reduction data directly.

        :param idx: The reduction index.
        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :return: The reduction data which normally is memory mapped to the
            underlying data, but will be copied on rollover.
        """
        total_length = self.length
        if stop - start > total_length:
            raise ValueError('requested size is too large')
        reduction = 1
        for i in range(idx + 1):
            reduction *= self.reductions[i].samples_per_step
        r_len = self.reductions[idx].length
        start = (start % total_length) // reduction
        stop = (stop % total_length) // reduction
        k = stop - start
        r = self.reductions_data[idx]
        if k == 0:
            return np.empty((0, 3, 4), dtype=np.float32)
        elif k < 0:  # copy on wrap
            k += r_len
            d = np.empty((k, 3, 4), dtype=np.float32)
            d[:(r_len - start), :, :] = r[start:, :, :]
            d[r_len - start:, :, :] = r[:stop, :, :]
            return d
        else:
            return r[start:stop, :, :]

    cdef _on_cbk(self, float * stats):
        if callable(self._callback):
            b = np.empty(12, dtype=np.float32)
            for i in range(12):
                b[i] = stats[i]
            b = b.reshape((3, 4))
            # todo handle variable sampling frequencies and reductions
            time_interval = 0.5  # seconds
            charge_picocoulomb = (b[0][0] * 1e12)  * time_interval
            energy_picojoules = (b[2][0] * 1e12) * time_interval
            if isfinite(charge_picocoulomb) and isfinite(energy_picojoules):
                self._charge_picocoulomb += int(charge_picocoulomb)
                self._energy_picojoules += int(energy_picojoules)
            charge = self._charge_picocoulomb * 1e-12
            energy = self._energy_picojoules * 1e-12
            statistic_names = ['mean', 'variance', 'min', 'max']
            data = {
                'time': {
                    'range': [0, time_interval],  # todo
                    'delta': time_interval,
                    'units': 's',
                },
                'signals': {
                    'current' : {
                        'statistics': _to_statistics(b[0, :]),
                        'units': 'A',
                        'integral_units': 'C',
                    },
                    'voltage' : {
                        'statistics': _to_statistics(b[1, :]),
                        'units': 'V',
                        'integral_units': '',
                    },
                    'power' : {
                        'statistics': _to_statistics(b[2, :]),
                        'units': 'W',
                        'integral_units': 'J',
                    },
                },
                'accumulators': {
                    'charge' : {
                        'units': 'C',
                        'value': charge,
                    },
                    'energy' : {
                        'value': energy,
                        'units': 'J',
                    },
                },
            }
            self._callback(data)


def _to_statistics(b):
    return {
        'μ': b[0],
        'σ2': b[1],
        'min': b[2],
        'max': b[3],
        'p2p': b[3] - b[2],
    }


cdef void _on_cbk(void * user_data, float * stats):
    cdef StreamBuffer self = <object> user_data
    self._on_cbk(stats)


def usb_packet_factory(packet_index, count=None):
    """Construct USB Bulk packets for testing.

    :param packet_index: The USB packet index for the first packet.
    :param count: The number of consecutive packets to construct.
    :return: The bytes containing the packet data
    """
    count = 1 if count is None else int(count)
    if count < 1:
        count = 1
    frame = np.empty(512 * count, dtype=np.uint8)
    for i in range(count):
        idx = packet_index + i
        k = i * 512
        frame[k + 0] = 1     # packet type raw
        frame[k + 1] = 0     # status = 0
        frame[k + 2] = 0x00  # length LSB
        frame[k + 3] = 0x02  # length MSB
        frame[k + 4] = idx & 0xff
        frame[k + 5] = (idx >> 8) & 0xff
        frame[k + 6] = 0
        frame[k + 7] = 0
        k += 8
        for j in range(126 * 2):
            v = (idx * 126 * 2 + j) << 2
            if j & 1:
                v |= j & 0x0002
            frame[k + j * 2] = v & 0xff
            frame[k + j * 2 + 1] = (v >> 8) & 0xff
    return frame


cpdef usb_packet_factory_signal(packet_index, count=None, samples_total=None):
    """Construct USB Bulk packets for testing.

    :param packet_index: The USB packet index for the first packet.
    :param count: The number of consecutive packets to construct.
    :param samples_total: The total number samples in the signal.  This value
        is used to unsure uniqueness.
    :return: The bytes containing the packet data
    """
    cdef uint16_t ij
    cdef uint16_t vj
    cdef float slope
    cdef uint64_t sample_offset = 0
    cdef int i
    cdef int j
    cdef int k
    cdef int z


    count = 1 if count is None else int(count)
    if count < 1:
        count = 1
    sample_rate = 2000000
    samples_total = sample_rate * 100 if samples_total is None else int(samples_total)
    slope = (2 ** 14 - 1) / samples_total
    stream_buffer = StreamBuffer(sample_rate // 10, [100])

    cdef frame = np.empty(512 * count, dtype=np.uint8)
    for i in range(count):
        packet_idx = packet_index + i
        k = i * 512
        frame[k + 0] = 1     # packet type raw
        frame[k + 1] = 0     # status = 0
        frame[k + 2] = 0x00  # length LSB
        frame[k + 3] = 0x02  # length MSB
        frame[k + 4] = packet_idx & 0xff
        frame[k + 5] = (packet_idx >> 8) & 0xff
        frame[k + 6] = 0
        frame[k + 7] = 0
        k += 8
        for j in range(SAMPLES_PER_PACKET):
            ij = <uint16_t> ((<float> sample_offset) * slope)
            vj = 16383 - ij
            ij = (ij << 2)
            vj = (vj << 2) | 0x02
            z = k + j * 4
            frame[z + 0] = ij & 0xff
            frame[z + 1] = (ij >> 8) & 0xff
            frame[z + 2] = vj & 0xff
            frame[z + 3] = (vj >> 8) & 0xff
    return frame


def stats_to_api(stats, t_start, t_stop):
    data = {
        'time': {
            'range': [t_start, t_stop],
            'delta': t_stop - t_start,
            'units': 's',  # seconds
        },
    }
    if stats is None:
        data['signals'] = {}
    else:
        data['signals'] = {
            'current': {
                'statistics': {},
                'units': 'A',  # ampere
                'integral_units': 'C',  # coulomb
            },
            'voltage': {
                'statistics': {},
                'units': 'V',  # volt
                'integral_units': None,
            },
            'power': {
                'statistics': {},
                'units': 'W',  # watt
                'integral_units': 'J',  # joule
            },
        }

        for i, field in enumerate(['current', 'voltage', 'power']):
            v = data['signals'][field]['statistics']
            v['μ'] = float(stats[i, 0])
            v['σ'] = float(np.sqrt(stats[i, 1]))
            v['min'] = float(stats[i, 2])
            v['max'] = float(stats[i, 3])
            v['p2p'] = v['max'] - v['min']

    return data