
# CHANGELOG

This file contains the list of changes made to pyjoulescope.


## 0.5.0

2019 Jul 26

*   Added bootloader_go to get matching device.
*   Added support for FW 1.1.0 JSON-only info format.
*   Added "config" option to Device initialization.
*   Added context manager to Device class.
*   Added file_replace module.
*   Added runtime check for Python 3.6+ and Python bits matching OS bits.
*   Moved current range change glitch suppression from FPGA to stream_buffer.
*   Improved macOS & Linux USB reliability, also be nice to libusb on removal.
*   Improved USB device error handling.
*   Removed NaN injection source.
*   Corrected invalid data possible on first 2 samples.
*   Added timeouts to bootloader / application transitions.
*   Added firmware_manager for more controller firmware update.
*   Made Device and Bootloader safe to open() when already open.


## 0.4.6

2019 Jul 15

*   Added optional application-specific metadata to datafile collection start.
*   Added support for voltage range to datafile: save/load/process correctly.
*   Improved missing sample (NaN) handling robustness.
*   Modified Driver.statistics_callback to match statistics_get format.
*   Compute total charge in addition to energy.
*   Fixed stream_buffer int/uint comparison warning.


## 0.4.5

2019 Jul 2

*   Added joulescope.inf to manually force Win 7 machines without WCID update 
    to recognize Joulescopes.
*   Fixed divide by zero error in stream_buffer.stats_compute_end.
*   Correctly close thread on device_thread open error.
*   Fixed potential 32 bit overflow issue in stream_buffer.
*   Fixed unchecked None data when not streaming in driver.View.


## 0.4.4

2019 Jun 28

*   Added "Quick Start" to README.md.
*   Added parameter "source" alias "on" for "raw".


## 0.4.3

2019 Jun 28

*   Added GPO alias values [0, 1].
*   Added GPO demo example.
*   Updated udev rules with instructions for improved security.
*   Increased Joulescope open timeout again, now 10.0 seconds from 6.0.
*   Fixed Joulescope Bootloader.go() to always close the bootloader.
*   Refactored so that "import joulescope" is useful.
*   Added "bootloaders_run_application".
*   Improved Device.bootloader error handling.
*   Removed libusb IOError if platform does not support hotplug.
*   Added no exception guarantee to scan() and bootloaders_run_application().
*   Fixed Cython compiler warnings.


## 0.4.2

2019 Jun 24

*   Improved error handling on Linux/Mac (libusb) device open.
*   Increase device open timeout.
*   Improved "capture" command logging and error handling.
*   Fixed string descriptor parsing for Linux/Mac (libusb).


## 0.4.1

2019 Jun 20

*   Added CREDITS.html file.


## 0.4.0

2019 Jun 20

*   Create a new ControlTransfer instance each time WinUsbDevice is opened.
*   Added StreamBuffer.stats_get to explicitly compute stats over a range.
*   Migrated to cython language_level=3.
*   Fixed error with statistics computation on NaN data, visible in UI as 
    min/max not being displayed correctly at some zoom levels.
*   Refactored statistics and added statistics_get to DataReader.
*   Unified View API for the physical device and recordings for UI.


## 0.3.1

2019 Jun 3

*   Added log messages for troubleshooting robustness issues.
*   Improved device thread error handling.
*   Eliminated small WinUsb memory leak in normal disconnect case.
*   Added progress callbacks for programming operations.
*   Fixed INFO record processing.
*   Added event_callback_fn and added support to win32 driver.
*   Improved win32 driver error handling & recovery.
*   Promoted driver._info to driver.info.
*   Improved documentation.
*   Keep x_max in view range whenever streaming is active.
*   Added view_time_to_sample_id.


## 0.3.0

2019 Apr 27

*   Added asynchronous control transfers so streaming continues correctly.
*   Improved robustness and recovery on Joulescope fw/hw issues.
*   Added GPI value read (IN) for compliance testing.


## 0.2.7

2019 Mar 2

*   Improved USB device error handling.
*   Allow data_recorder raw defaults to fetch entire file.
*   Added VERSION and __version__ members.
*   Added support for older Mac OS versions when packaged.


## 0.2.6

2019 Feb 16

*   Fixed incorrect column index for "power"


## 0.2.4

2019 Feb 10

*   Fixed incorrect formatting in three_sig_figs for negative numbers.


## 0.2.3

2019 Feb 8

*   Modified span scaling to use pivot point rather than force to center,
    which results in more intuitive UI behavior.
*   Suppress glitches (up to 2 samples) which occur on current range switches.


## 0.2.2

2019 Jan 27

*   Added data file read & data file write support (file format changed).
*   Added command-line tool "recording" for dealing the ".jls" files.


## 0.2.1

2019 Jan 25

*   Added linux support using libusb
*   Added Mac OS X support using libusb


## 0.1.5

2018 Dec 5

*   Check status during sensor firmware programming (was presuming success!).
*   Fixed max range when value is always negative.


## 0.1.0

2018 Oct 9

*   Initial public release.
