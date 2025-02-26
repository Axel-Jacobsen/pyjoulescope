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

from joulescope.usb import core as usb_core
from joulescope.usb.impl_tools import RunUntilDone
from joulescope.usb.core import SetupPacket, ControlTransferResponse
from joulescope.usb.scan_info import INFO
from typing import List
import time
import threading
from contextlib import contextmanager
import platform
import numpy as np
import os
import sys
import struct
import ctypes
import ctypes.util
from ctypes import Structure, c_uint8, c_uint16, c_uint32, c_uint, \
    c_int, c_char, c_ssize_t, c_void_p, POINTER, pointer, byref
import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


STRING_LENGTH_MAX = 255
CONTROL_TRANSFER_TIMEOUT_MS = 1000  # default in milliseconds

find_lib = ctypes.util.find_library('usb-1.0')
if find_lib is None:
    if platform.system() == 'Darwin' and getattr(sys, 'frozen', False):
        os_version = platform.uname().release.split('.')[0]
        find_lib = os.path.join(sys._MEIPASS, '%s_libusb-1.0.0.dylib' % os_version)
        log.info('Darwin lib: %s', find_lib)
    else:
        raise RuntimeError('Could not import libusb')
_lib = ctypes.cdll.LoadLibrary(find_lib)


class DescriptorType:
    DEVICE = 0x01
    CONFIG = 0x02
    STRING = 0x03
    INTERFACE = 0x04
    ENDPOINT = 0x05
    BOS = 0x0f
    DEVICE_CAPABILITY = 0x10
    HID = 0x21
    REPORT = 0x22
    PHYSICAL = 0x23
    HUB = 0x29
    SUPERSPEED_HUB = 0x2a
    SS_ENDPOINT_COMPANION = 0x30


class TransferType:
    CONTROL = 0
    ISOCHRONOUS = 1
    BULK = 2
    INTERRUPT = 3
    BULK_STREAM = 4


class TransferStatus:
    COMPLETED = 0
    ERROR = 1
    TIMED_OUT = 2
    CANCELLED = 3
    STALL = 4
    NO_DEVICE = 5
    OVERFLOW = 6


class TransferFlags:
    SHORT_NOT_OK = 1 << 0
    FREE_BUFFER = 1 << 1
    FREE_TRANSFER = 1 << 2
    ADD_ZERO_PACKET = 1 << 3


class ReturnCodes:
    SUCCESS = 0
    ERROR_IO = -1
    ERROR_INVALID_PARAM = -2
    ERROR_ACCESS = -3
    ERROR_NO_DEVICE = -4
    ERROR_NOT_FOUND = -5
    ERROR_BUSY = -6
    ERROR_TIMEOUT = -7
    ERROR_OVERFLOW = -8
    ERROR_PIPE = -9
    ERROR_INTERRUPTED = -10
    ERROR_NO_MEM = -11
    ERROR_NOT_SUPPORTED = -12
    ERROR_OTHER = -99


class _libusb_device_descriptor(Structure):
    _fields_ = [
        ('bLength', c_uint8),
        ('bDescriptorType', c_uint8),
        ('bcdUSB', c_uint16),
        ('bDeviceClass', c_uint8),
        ('bDeviceSubClass', c_uint8),
        ('bDeviceProtocol', c_uint8),
        ('bMaxPacketSize0', c_uint8),
        ('idVendor', c_uint16),
        ('idProduct', c_uint16),
        ('bcdDevice', c_uint16),
        ('iManufacturer', c_uint8),
        ('iProduct', c_uint8),
        ('iSerialNumber', c_uint8),
        ('bNumConfigurations', c_uint8)]


# typedef void (LIBUSB_CALL *libusb_transfer_cb_fn)(struct libusb_transfer *transfer);
libusb_transfer_cb_fn = ctypes.CFUNCTYPE(None, c_void_p)

class _libusb_transfer(Structure):
    _fields_ = [
        ('dev_handle', c_void_p),
        ('flags', c_uint8),
        ('endpoint_id', c_uint8),
        ('endpoint_type', c_uint8),
        ('timeout_ms', c_uint),
        ('status', c_uint),
        ('length', c_int),
        ('actual_length', c_int),
        ('callback', libusb_transfer_cb_fn),
        ('user_data', c_void_p),
        ('buffer', POINTER(c_uint8)),
        ('num_iso_packets', c_int),
        # struct libusb_iso_packet_descriptor iso_packet_desc[ZERO_SIZED_ARRAY];
    ]

# typedef struct libusb_context libusb_context; - c_void_p
# typedef struct libusb_device libusb_device; - c_void_p
# typedef struct libusb_device_handle libusb_device_handle; c_void_p

# int LIBUSB_CALL libusb_init(libusb_context **ctx);
_lib.libusb_init.restype = c_int
_lib.libusb_init.argtypes = [POINTER(c_void_p)]

# void LIBUSB_CALL libusb_exit(libusb_context *ctx);
_lib.libusb_exit.restype = None
_lib.libusb_exit.argtypes = [c_void_p]

# ssize_t LIBUSB_CALL libusb_get_device_list(libusb_context *ctx,
#	libusb_device ***list);
_lib.libusb_get_device_list.restype = c_ssize_t
_lib.libusb_get_device_list.argtypes = [c_void_p, POINTER(POINTER(c_void_p))]


# void LIBUSB_CALL libusb_free_device_list(libusb_device **list,
#	int unref_devices);
_lib.libusb_free_device_list.restype = None
_lib.libusb_free_device_list.argtypes = [POINTER(c_void_p), c_int]


# int LIBUSB_CALL libusb_open(libusb_device *dev, libusb_device_handle **dev_handle);
_lib.libusb_open.restype = c_int
_lib.libusb_open.argtypes = [c_void_p, POINTER(c_void_p)]

# void LIBUSB_CALL libusb_close(libusb_device_handle *dev_handle);
_lib.libusb_close.restype = None
_lib.libusb_close.argtypes = [c_void_p]

# int LIBUSB_CALL libusb_set_configuration(libusb_device_handle *dev_handle,
#	int configuration);
_lib.libusb_set_configuration.restype = c_int
_lib.libusb_set_configuration.argtypes = [c_void_p, c_int]

# int LIBUSB_CALL libusb_claim_interface(libusb_device_handle *dev_handle,
#	int interface_number);
_lib.libusb_claim_interface.restype = c_int
_lib.libusb_claim_interface.argtypes = [c_void_p, c_int]

# int LIBUSB_CALL libusb_release_interface(libusb_device_handle *dev_handle,
#	int interface_number);
_lib.libusb_release_interface.restype = c_int
_lib.libusb_release_interface.argtypes = [c_void_p, c_int]

# int LIBUSB_CALL libusb_set_interface_alt_setting(libusb_device_handle *dev_handle,
#	int interface_number, int alternate_setting);
_lib.libusb_set_interface_alt_setting.restype = c_int
_lib.libusb_set_interface_alt_setting.argtypes = [c_void_p, c_int, c_int]


# int LIBUSB_CALL libusb_get_device_descriptor(libusb_device *dev,
#	struct libusb_device_descriptor *desc);
_lib.libusb_get_device_descriptor.restype = c_int
_lib.libusb_get_device_descriptor.argtypes = [c_void_p, POINTER(_libusb_device_descriptor)]

# int LIBUSB_CALL libusb_control_transfer(libusb_device_handle *dev_handle,
#	uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex,
#	unsigned char *data, uint16_t wLength, unsigned int timeout);
_lib.libusb_control_transfer.restype = c_int
_lib.libusb_control_transfer.argtypes = [c_void_p, c_uint8, c_uint8, c_uint16, c_uint16,
                                         POINTER(c_uint8), c_uint16, c_int]

# struct libusb_transfer * LIBUSB_CALL libusb_alloc_transfer(int iso_packets);
_lib.libusb_alloc_transfer.restype = POINTER(_libusb_transfer)
_lib.libusb_alloc_transfer.argtypes = [c_int]

# int LIBUSB_CALL libusb_submit_transfer(struct libusb_transfer *transfer);
_lib.libusb_submit_transfer.restype = c_int
_lib.libusb_submit_transfer.argtypes = [POINTER(_libusb_transfer)]

# int LIBUSB_CALL libusb_cancel_transfer(struct libusb_transfer *transfer);
_lib.libusb_cancel_transfer.restype = c_int
_lib.libusb_cancel_transfer.argtypes = [POINTER(_libusb_transfer)]

# void LIBUSB_CALL libusb_free_transfer(struct libusb_transfer *transfer);
_lib.libusb_free_transfer.restype = None
_lib.libusb_free_transfer.argtypes = [POINTER(_libusb_transfer)]


class TimeVal(Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_usec", ctypes.c_long)
    ]

# int LIBUSB_CALL libusb_handle_events_timeout(libusb_context *ctx,
#	struct timeval *tv);
_lib.libusb_handle_events_timeout.restype = c_int
_lib.libusb_handle_events_timeout.argtypes = [c_void_p, POINTER(TimeVal)]

# int LIBUSB_CALL libusb_handle_events(libusb_context *ctx)
_lib.libusb_handle_events.restype = c_int
_lib.libusb_handle_events.argtypes = [c_void_p]


class HotplugFlag:
    NONE = 0
    ENUMERATE = 1 << 0


class HotplugEvent:
    DEVICE_ARRIVED = 0x01
    DEVICE_LEFT    = 0x02


HOTPLUG_MATCH_ANY = -1


# typedef int (LIBUSB_CALL *libusb_hotplug_callback_fn)(libusb_context *ctx,
#						libusb_device *device,
#						libusb_hotplug_event event,
#						void *user_data);
_libusb_hotplug_callback_fn = ctypes.CFUNCTYPE(c_int, c_void_p, c_void_p, c_int, c_void_p)

# int LIBUSB_CALL libusb_hotplug_register_callback(libusb_context *ctx,
#						libusb_hotplug_event events,
#						libusb_hotplug_flag flags,
#						int vendor_id, int product_id,
#						int dev_class,
#						libusb_hotplug_callback_fn cb_fn,
#						void *user_data,
#						libusb_hotplug_callback_handle *callback_handle);
_lib.libusb_hotplug_register_callback.restype = c_int
_lib.libusb_hotplug_register_callback.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int,
                                                  _libusb_hotplug_callback_fn, c_void_p, POINTER(c_int)]

# void LIBUSB_CALL libusb_hotplug_deregister_callback(libusb_context *ctx,
#						libusb_hotplug_callback_handle callback_handle);
_lib.libusb_hotplug_deregister_callback.restype = c_int
_lib.libusb_hotplug_deregister_callback.argtypes = [c_void_p, c_int]


class Capability:
    HAS_CAPABILITY = 0x0000
    HAS_HOTPLUG = 0x0001
    HAS_HID_ACCESS = 0x0100
    SUPPORTS_DETACH_KERNEL_DRIVER = 0x0101


# int LIBUSB_CALL libusb_has_capability(uint32_t capability);
_lib.libusb_has_capability.restype = c_int
_lib.libusb_has_capability.argtypes = [c_uint32]



def _libusb_context_create():
    ctx = c_void_p()

    rc = _lib.libusb_init(pointer(ctx))
    if rc:
        raise RuntimeError('Could not open libusb')
    return ctx


def _libusb_context_destroy(ctx):
    _lib.libusb_exit(ctx)


@contextmanager
def _libusb_context():
    ctx = _libusb_context_create()
    try:
        yield ctx
    finally:
        _libusb_context_destroy(ctx)


def _path_split(path):
    vid, pid, serial_number = path.split('/')
    return int(vid, 16), int(pid, 16), serial_number


def _get_string_descriptor(device, index):
    request_type = usb_core.RequestType(direction='in', type_='standard', recipient='device').u8
    byte_buffer = bytearray(STRING_LENGTH_MAX)
    buffer_type = c_uint8 * STRING_LENGTH_MAX
    buffer = buffer_type.from_buffer(byte_buffer)

    # determine default language
    rv = _lib.libusb_control_transfer(device, request_type, usb_core.Request.GET_DESCRIPTOR,
                                      (DescriptorType.STRING << 8), 0,
                                      buffer, STRING_LENGTH_MAX,
                                      1000)
    if rv < 0:
        raise RuntimeError('control_transfer could not get language: %d' % (rv, ))
    langid = int(byte_buffer[2]) | (int(byte_buffer[3]) << 8)

    rv = _lib.libusb_control_transfer(device, request_type, usb_core.Request.GET_DESCRIPTOR,
                                      (DescriptorType.STRING << 8) | (index & 0xff), langid,
                                      buffer, STRING_LENGTH_MAX,
                                      1000)

    if rv < 0:
        raise RuntimeError('control transfer could not get string descriptor: %d' % (rv, ))
    buffer_len = min(rv, byte_buffer[0])
    # byte 0 is length, byte 1 is string identifier
    return byte_buffer[2:buffer_len].decode('UTF-16-LE')


_transfer_callback_discard_fn = libusb_transfer_cb_fn(lambda x: None)
"""Default null callback that is always safe."""


class Transfer:

    def __init__(self, size):
        try:
            self.size = len(size)  # also serves as list-like duck-typing test
            self.buffer = np.frombuffer(size, dtype=np.uint8)
            log.debug('Transfer: copy buffer %d', self.size)
        except TypeError:
            self.size = size
            self.buffer = np.empty(self.size, dtype=np.uint8)
            log.debug('Transfer: create buffer %d', self.size)
        self.transfer = _lib.libusb_alloc_transfer(0)  # type: _libusb_transfer
        self.addr = ctypes.addressof(self.transfer.contents)
        transfer = self.transfer[0]
        self.buffer_ptr = self.buffer.ctypes.data_as(POINTER(c_uint8))
        transfer.buffer = self.buffer_ptr
        transfer.flags = 0
        transfer.length = self.size
        transfer.actual_length = 0
        transfer.user_data = None
        transfer.num_iso_packets = 0
        transfer.status = TransferStatus.COMPLETED
        transfer.timeout_ms = 1000  # milliseconds
        transfer.callback = _transfer_callback_discard_fn

    def __del__(self):
        _lib.libusb_free_transfer(self.transfer)


class ControlTransferAsync:

    def __init__(self, handle, remove_fn):
        """Manage asynchronous control transfers.

        :param handle: The device handle.
        :param remove_fn: The function() called on device removal detection.
        """
        self._handle = handle
        self._transfer_callback_fn = libusb_transfer_cb_fn(self._transfer_callback)
        self._commands = []  # Pending control transfer commands as list of [cbk_fn, setup_packet, buffer]
        self._transfer_pending = None  # type: Transfer
        self._time_start = None
        self._remove_fn = remove_fn

    def __str__(self):
        return 'ControlTransferAsync()'

    def _transfer_callback(self, transfer_void_ptr):
        if self._transfer_pending is None:
            log.warning('Transfer callback when none pending')
            return
        if self._transfer_pending.addr != transfer_void_ptr:
            log.warning('Transfer mismatch')
            return
        transfer, self._transfer_pending = self._transfer_pending, None
        if self._commands:
            self._finish(self._commands.pop(0), transfer)
        else:
            log.warning('Transfer callback when no commands')
        self._issue()

    def close(self):
        commands, self._commands = self._commands, []
        if self._transfer_pending:
            log.info('ControlTransferAsync.close cancel pending transfer, %d', len(commands))
            transfer, self._transfer_pending = self._transfer_pending, None
            transfer.transfer[0].callback = _transfer_callback_discard_fn
            _lib.libusb_cancel_transfer(transfer.transfer)
            # callback function will be invoked later
        else:
            log.info('ControlTransferAsync.close %d', len(commands))
        for cbk_fn, setup_packet, _ in commands:
            try:
                response = usb_core.ControlTransferResponse(setup_packet, TransferStatus.CANCELLED, None)
                cbk_fn(response)
            except Exception:
                log.exception('while closing in callback')

    def pend(self, cbk_fn, setup_packet: usb_core.SetupPacket, buffer=None):
        """Pend an asynchronous Control Transfer.

        :param cbk_fn: The function to call when the control transfer completes.
            A :class:`usb_core.ControlTransferResponse` is the sole argument.
        :param setup_packet:
        :param buffer: The buffer (if length > 0) for write transactions.
        """
        command = [cbk_fn, setup_packet, buffer]
        was_empty = not bool(self._commands)
        self._commands.append(command)
        if was_empty:
            return self._issue()
        return True

    def _issue(self):
        if not self._commands:
            return True
        log.debug('preparing')
        _, setup_packet, buffer = self._commands[0]
        hdr = struct.pack('<BBHHH', setup_packet.request_type, setup_packet.request,
                          setup_packet.value, setup_packet.index, setup_packet.length)
        if buffer is not None:
            transfer = Transfer(hdr + buffer)
        else:
            transfer = Transfer(len(hdr) + setup_packet.length)
            transfer.buffer[:len(hdr)] = np.frombuffer(hdr, dtype=np.uint8)
        t = transfer.transfer[0]
        t.dev_handle = self._handle
        t.endpoint_id = 0
        t.endpoint_type = TransferType.CONTROL
        t.callback = self._transfer_callback_fn
        self._transfer_pending = transfer
        self._time_start = time.time()
        rv = _lib.libusb_submit_transfer(transfer.transfer)
        if 0 == rv:
            log.debug('libusb_submit_transfer [control]')
        else:
            log.warning('libusb_submit_transfer [control] => %d', rv)
            if t.status == 0:
                if rv == ReturnCodes.ERROR_NO_DEVICE:
                    t.status = TransferStatus.NO_DEVICE
                    self._remove_fn()
                else:
                    t.status = TransferStatus.ERROR
            self._transfer_callback(transfer.addr)
            return False
        return True

    def _finish(self, command, transfer):
        buffer = None
        rc = transfer.transfer[0].status
        cbk_fn, setup_packet, _ = command
        pkt = usb_core.RequestType(value=setup_packet.request_type)
        duration = time.time() - self._time_start
        if rc == TransferStatus.NO_DEVICE:
            log.warning('device_removed')
            self._remove_fn()
        if pkt.direction == 'out':
            log.debug('ControlTransferAsync._finish rc=%d, duration=%.6f s', rc, duration)
        else:
            actual_length = transfer.transfer[0].actual_length
            log.debug('ControlTransferAsync._finish rc=%d, duration=%.6f s, length: %s, %s',
                      rc, duration, setup_packet.length, actual_length)
            buffer = bytes(transfer.buffer[8:(actual_length+8)])
        response = usb_core.ControlTransferResponse(setup_packet, rc, buffer)
        cbk_fn(response)


class EndpointIn:

    def __init__(self, handle, pipe_id, transfers, block_size, data_fn, process_fn, remove_fn):
        """Manage an in endpoint.

        :param handle: The device handle.
        :param pipe_id: The endpoint IN pipe identifier.
        :param transfers: The number of outstanding transfers to pend.
        :param block_size: The size of each transfer in bytes.
        :param data_fn: The function to call with the received endpoint IN data.
            After the last block, data_fn is called with None to indicate the
            last transfer.  The data_fn should normally return True, but can
            return False to stop the endpoint streaming.
        :param process_fn: The function() called after data_fn was called.
            This function can have more latency than data_fn.
        :param remove_fn: The function() call on device removal.
        """
        self._handle = handle
        self.pipe_id = pipe_id  # (endpoint_id & 0x7f) | 0x80
        self._config = {
            'transfer_count': transfers,
            'transfer_size_bytes': (block_size + 511 // 512)
        }
        self._data_fn = data_fn
        self._process_fn = process_fn
        self._remove_fn = remove_fn

        self._state = self.ST_IDLE
        self._time_last = None
        self._transfers_free = []  # Transfer
        self._transfers_pending = []  # Transfer
        self.transfers_processed = 0

        self.transfer_count = 0
        self.byte_count_window = 0
        self.byte_count_total = 0
        self._transfer_callback_fn = libusb_transfer_cb_fn(self._transfer_callback)
        self._init()

    ST_IDLE = 0
    ST_RUNNING = 1
    ST_STOPPING = 2
    ST_ABORT = 3

    def __str__(self):
        return 'EndpointIn(0x%02x)' % (self.pipe_id, )

    def _transfer_pending_pop(self, transfer_void_ptr):
        for idx in range(len(self._transfers_pending)):
            if transfer_void_ptr == self._transfers_pending[idx].addr:
                return self._transfers_pending.pop(idx)
        log.warning('%s _transfer_pending_pop not found', self)
        raise IOError('%s _transfer_pending_pop not found' % (self, ))

    def _transfer_callback(self, transfer_void_ptr):
        transfer = self._transfer_pending_pop(transfer_void_ptr)
        self.transfer_count += 1
        t = transfer.transfer[0]

        try:
            if self._state == self.ST_RUNNING:
                if t.status == TransferStatus.COMPLETED:
                    self.byte_count_window += t.actual_length
                    self.byte_count_total += t.actual_length
                    self.transfers_processed += 1
                    buffer = transfer.buffer[:t.actual_length]
                    if self._data_fn(buffer):
                        log.info('%s terminated by data_fn', self)
                        self._cancel()
                elif t.status == TransferStatus.NO_DEVICE:
                    log.warning('%s, transfer callback with status NO DEVICE', self)
                    self._remove_fn()
                    self._cancel()
                else:
                    log.warning('%s, transfer callback with status %d', self, t.status)
        finally:
            self._transfers_free.append(transfer)
        self._pend()
        if self._state in [self.ST_STOPPING, self.ST_ABORT]:
            if 0 == len(self._transfers_pending):
                self._data_fn(None)  # indicate done with None
                self._state = self.ST_IDLE
                log.info('%s stop => idle', self)
            else:
                log.debug('awaiting transfer completion')

    def _init(self):
        for i in range(self._config['transfer_count']):
            transfer = Transfer(self._config['transfer_size_bytes'])
            t = transfer.transfer[0]
            t.dev_handle = self._handle
            t.endpoint_id = self.pipe_id
            t.endpoint_type = TransferType.BULK
            t.callback = self._transfer_callback_fn
            self._transfers_free.append(transfer)

    def _pend(self):
        while self._state == self.ST_RUNNING and len(self._transfers_free):
            transfer = self._transfers_free.pop(0)
            transfer.transfer[0].status = TransferStatus.COMPLETED
            rv = _lib.libusb_submit_transfer(transfer.transfer)
            if rv:
                self._transfers_free.append(transfer)
                if rv in [ReturnCodes.ERROR_BUSY]:
                    log.info('libusb_submit_transfer busy')
                else:  # no device, not supported, or other error
                    log.error('libusb_submit_transfer => %d  ABORT', rv)
                    self._cancel()
                    self._remove_fn()
                    self._state = self.ST_ABORT
                break  # give system time to recover
            else:
                self._transfers_pending.append(transfer)

    def _cancel(self):
        self._state = self.ST_STOPPING
        log.info('%s cancel %d', self, len(self._transfers_pending))
        for transfer in self._transfers_pending:
            _lib.libusb_cancel_transfer(transfer.transfer)

    def process_signal(self):
        if self.transfer_count and self._state == self.ST_RUNNING:
            self.transfer_count = 0
            if self._process_fn():
                log.info('%s terminated by process_fn', self)
                self._cancel()
                return True
        return False

    def start(self):
        log.info("%s start transfer size = %d bytes" % (self, self._config['transfer_size_bytes']))
        self.transfer_count = 0
        self.byte_count_window = 0
        self.byte_count_total = 0
        self._state = self.ST_RUNNING
        self._time_last = time.time()
        self._pend()
        time.sleep(0.0001)

    def stop(self):
        if self._state != self.ST_IDLE:
            log.info('%s stop', self)
            self._cancel()

    def status(self):
        """Get the endpoint status.

        :return: A dict mapping status name to a value..  The value is a dict
            containing 'value' and 'units'.
        """
        time_now = time.time()
        duration = time_now - self._time_last
        if duration < 0.01:
            throughput = 0.0
        else:
            throughput = self.byte_count_window / duration
        status = {
            'bytes': {'value': self.byte_count_total, 'units': 'bytes'},
            'transfers': {'value': self.transfer_count, 'units': 'transfers'},
            'duration': {'value': duration, 'units': 'seconds'},
            'throughput': {'value': throughput, 'units': 'Bps'},
        }
        self.byte_count_window = 0
        self._time_last = time_now
        return status


def may_raise_ioerror(rv, msg):
    if 0 != rv:
        s = msg + (' [%d]' % (rv, ))
        log.warning(s)
        raise IOError(s)
    else:
        log.debug('%s: success', msg.split(' ')[0])


class LibUsbDevice:
    """The LibUSB :class:`usb.api.Device` implementation"""

    def __init__(self, path):
        self._ctx = None
        self._path = path
        self._handle = c_void_p(None)
        self._endpoints = {}
        self._control_transfer = None  # type: ControlTransferAsync
        self._removed = False

    def __str__(self):
        return f'Joulescope {self.serial_number}'

    @property
    def serial_number(self):
        return self._path.split('/')[-1]

    def on_removed(self):
        log.info('Device removal signaled')
        self._removed = True

    def _open(self):
        log.info('open: start %s', self._path)
        self._ctx = _libusb_context_create()
        descriptor = _libusb_device_descriptor()
        devices = POINTER(c_void_p)()
        vid, pid, serial_number = _path_split(self._path)
        sz = _lib.libusb_get_device_list(self._ctx, pointer(devices))
        try:
            for idx in range(sz):
                device = devices[idx]
                if _lib.libusb_get_device_descriptor(device, pointer(descriptor)):
                    continue
                if vid == descriptor.idVendor and pid == descriptor.idProduct:
                    dh = c_void_p(None)
                    rv = _lib.libusb_open(device, dh)
                    if rv < 0:
                        log.info('Could not open device: %04x/%04x', vid, pid)
                        continue
                    if serial_number == _get_string_descriptor(dh, descriptor.iSerialNumber):
                        self._handle = dh
                        log.info('open: success')
                        return
            log.warning('open:failed')
            raise IOError('open:failed')
        finally:
            _lib.libusb_free_device_list(devices, 0)

    def open(self, event_callback_fn=None):
        # todo support event_callback_fn on errors
        self.close()
        try:
            self._open()
            log.info('open: configure device')
            rv = _lib.libusb_set_configuration(self._handle, 1)
            may_raise_ioerror(rv, 'libusb_set_configuration 1 failed')
            rv = _lib.libusb_claim_interface(self._handle, 0)
            may_raise_ioerror(rv, 'libusb_claim_interface 0 failed')
            rv = _lib.libusb_set_interface_alt_setting(self._handle, 0, 0)
            may_raise_ioerror(rv, 'libusb_set_interface_alt_setting 0,0 failed')
            self._control_transfer = ControlTransferAsync(self._handle, self.on_removed)
            log.info('open: done')
        except IOError:
            log.exception('open failed: %s', self._path)
            self.close()
            raise
        except Exception as ex:
            log.exception('open failed: %s', self._path)
            self.close()
            raise IOError(ex)

    def close(self):
        log.info('close')
        if self._control_transfer:
            self._control_transfer.close()
            self._control_transfer = None
        if self._handle:
            _lib.libusb_close(self._handle)
            self._handle = c_void_p(None)
        if self._ctx:
            _libusb_context_destroy(self._ctx)
            self._ctx = None

    def control_transfer_out(self, cbk_fn, recipient, type_, request, value=0, index=0, data=None):
        if cbk_fn is None:
            run_until_done = RunUntilDone(1.0, 'control_transfer_out')
            self.control_transfer_out(run_until_done.cbk_fn, recipient, type_, request, value, index, data)
            while not run_until_done.is_done():
                self.process(0.01)
            return run_until_done.value_args0
        request_type = usb_core.RequestType(direction='out', type_=type_, recipient=recipient).u8
        length = 0 if data is None else len(data)
        setup_packet = usb_core.SetupPacket(request_type, request, value, index, length)
        return self._control_transfer.pend(cbk_fn, setup_packet, data)

    def control_transfer_in(self, cbk_fn, recipient, type_, request, value, index, length) -> ControlTransferResponse:
        if cbk_fn is None:
            run_until_done = RunUntilDone(1.0, 'control_transfer_in')
            self.control_transfer_in(run_until_done.cbk_fn, recipient, type_, request, value, index, length)
            while not run_until_done.is_done():
                self.process(0.01)
            return run_until_done.value_args0
        request_type = usb_core.RequestType(direction='in', type_=type_, recipient=recipient).u8
        setup_packet = usb_core.SetupPacket(request_type, request, value, index, length)
        return self._control_transfer.pend(cbk_fn, setup_packet, None)

    def read_stream_start(self, endpoint_id, transfers, block_size, data_fn, process_fn):
        pipe_id = (endpoint_id & 0x7f) | 0x80
        endpoint = self._endpoints.pop(pipe_id, None)
        if endpoint is not None:
            log.warning('repeated start')
            endpoint.stop()
        endpoint = EndpointIn(self._handle, pipe_id, transfers,
                              block_size, data_fn, process_fn, self.on_removed)
        self._endpoints[endpoint.pipe_id] = endpoint
        endpoint.start()

    def read_stream_stop(self, endpoint_id):
        log.info('read_stream_stop %d', endpoint_id)
        pipe_id = (endpoint_id & 0x7f) | 0x80
        endpoint = self._endpoints.pop(pipe_id, None)
        if endpoint is not None:
            endpoint.stop()

    def status(self):
        e = {}
        s = {'endpoints': e}
        for endpoint in self._endpoints.values():
            e[endpoint.pipe_id] = endpoint.status()
        return s

    def signal(self):
        pass  # todo, currently delays in process for up to 25 ms waiting for libusb_handle_events_timeout

    def process(self, timeout=None):
        if self._ctx and not self._removed:
            timeval = TimeVal(tv_sec=0, tv_usec=25000)
            _lib.libusb_handle_events_timeout(self._ctx, byref(timeval))
            endpoints_stop = []
            for endpoint in self._endpoints.values():
                endpoint.process_signal()
                if endpoint._state == endpoint.ST_IDLE:
                    endpoints_stop.append(endpoint.pipe_id)
            for pipe_id in endpoints_stop:
                self.read_stream_stop(pipe_id)
        else:
            pass  # time.sleep(0.001)


class DeviceNotify:

    def __init__(self, cbk):
        self._cbk = cbk
        self._window_thread = None
        self._do_quit = True
        if 0 == _lib.libusb_has_capability(Capability.HAS_HOTPLUG):
            log.warning('libusb does not support hotplug')
            return
        self.open()

    def _hotplug_cbk(self, ctx, device, ev, user_data):
        inserted = bool(ev & HotplugEvent.DEVICE_ARRIVED)
        removed = bool(ev & HotplugEvent.DEVICE_LEFT)
        log.info('hotplug: inserted=%s, removed=%s', inserted, removed)
        self._cbk(inserted, 0)
        return 0

    def _run(self):
        if 0 == _lib.libusb_has_capability(Capability.HAS_HOTPLUG):
            log.warning('libusb does not support hotplug')
            # todo revert to polling method?
            return
        log.debug('_run_window start')
        timeval = TimeVal(tv_sec=0, tv_usec=100000)
        timeval_ptr = pointer(timeval)
        handle = c_int()
        cbk_fn = _libusb_hotplug_callback_fn(self._hotplug_cbk)
        cbk_user_data = c_void_p()
        with _libusb_context() as ctx:
            rv = _lib.libusb_hotplug_register_callback(
                ctx,
                HotplugEvent.DEVICE_ARRIVED | HotplugEvent.DEVICE_LEFT,
                0, # flags
                HOTPLUG_MATCH_ANY,  # vid
                HOTPLUG_MATCH_ANY,  # pid
                HOTPLUG_MATCH_ANY,  # device class
                cbk_fn,
                cbk_user_data,
                byref(handle))
            if rv:
                raise IOError('could not register hotplug')
            while not self._do_quit:
                _lib.libusb_handle_events_timeout(ctx, timeval_ptr)
            _lib.libusb_hotplug_deregister_callback(ctx, handle)

    def open(self):
        self.close()
        self._do_quit = False
        log.info('open')
        self._window_thread = threading.Thread(name='device_notify', target=self._run)
        self._window_thread.start()

    def close(self):
        if self._window_thread:
            log.info('close')
            self._do_quit = True
            self._window_thread.join()
            self._window_thread = None


def scan(name: str) -> List[LibUsbDevice]:
    """Scan for attached devices.

    :param name: The case-insensitive name of the device to scan.
    :return: The list of discovered :class:`WinUsbDevice` instances.
    """
    with _libusb_context() as ctx:
        paths = []
        infos = INFO[name.lower()]
        descriptor = _libusb_device_descriptor()
        devices = POINTER(c_void_p)()
        sz = _lib.libusb_get_device_list(ctx, pointer(devices))
        try:
            for idx in range(sz):
                device = devices[idx]
                if _lib.libusb_get_device_descriptor(device, pointer(descriptor)):
                    raise RuntimeError('descriptor')
                for info in infos:
                    vid = info['vendor_id']
                    pid = info['product_id']
                    if vid == descriptor.idVendor and pid == descriptor.idProduct:
                        dh = c_void_p(None)
                        rv = _lib.libusb_open(device, pointer(dh))
                        if rv < 0:
                            log.info('Could not open device: %04x/%04x', vid, pid)
                            continue
                        try:
                            serial_number = _get_string_descriptor(dh, descriptor.iSerialNumber)
                        finally:
                            _lib.libusb_close(dh)
                        path = '%04x/%04x/%s' % (vid, pid, serial_number)
                        paths.append(path)
        finally:
            _lib.libusb_free_device_list(devices, 0)

        if not len(paths):
            log.info('scan found no devices')
            return []
        log.info('scan found %s' % paths)
        devices = [LibUsbDevice(p) for p in paths]
        return devices


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    devices = scan('Joulescope')
    print('\n'.join([str(d) for d in devices]))
    if len(devices):
        d = devices[0]
        d.open()
        rv = d.control_transfer_in('device', 'vendor', request=4,
            value=0, index=0, length=128)
        print(rv)
        d.close()
