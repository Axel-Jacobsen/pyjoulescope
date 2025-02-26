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

from joulescope.paths import JOULESCOPE_DIR
from joulescope.driver import scan, scan_require_one, scan_for_changes, \
    bootloaders_run_application, bootloader_go
import sys
import platform

try:
    from .version import VERSION
except ImportError:
    VERSION = 'UNRELEASED'

__version__ = VERSION


__all__ = [scan, scan_require_one, scan_for_changes, bootloaders_run_application,
           bootloader_go,
           JOULESCOPE_DIR, VERSION, __version__]

if sys.hexversion < 0x030600:
    raise RuntimeError('joulescope requires Python 3.6+ 64-bit')


# Although only 64-bit OS/Python is supported, may be able to run on 32bit Python / 32bit Windows.
p_sz, _ = platform.architecture()
is_32bit = sys.maxsize < (1 << 32)

if (is_32bit and '32' not in p_sz) or (not is_32bit and '64' not in p_sz):
    raise RuntimeError('joulescope Python bits must match platform bits')
