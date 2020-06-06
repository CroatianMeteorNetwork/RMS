# Devices


## UTV007
kernel driver
USB ID: 2002
usage:
 1. mplayer tv:// -tv device=/dev/video0
 2. quit mplayer
 3. OpenCV
restrictions:
 - only 720x576@25fps

Brightness/contrast/saturation/hue of usbtv kernel module can be controlled by v4l2-ctl (from v4l-utils) with 
e.g. v4l2-ctl --set-ctrl=brightness=512. Default values in Ubuntu 16.04 are too dark by default.

## Somagic
userspace module
USB ID(non-initized): 1c88:0007
USB ID(initized): 1c88:003f
usage:
 1. sudo somagic-init
	errors (exit code 1):
		USB device already initialized
		Failed to open USB device: Permission denied
		USB device 1c88:0007 was not found. Is the device attached?
	success = exit code 0
 2. feed somagic-capture over standard input/output
restrictions:
 - has to use binary-blob (firmware)... can be extracted over USB, extracted from CD or downloaded illegally (propertary)
 - no V4L2 driver, unless kernel is recompiled (no way!)

## Arkmicro UVC
UVC-compatible (ie. kernel driver)
USB ID: 18ec:5850
usage:
 1. OpenCV
restrictions:
 - only 640x480 (which fps?)
 - seems that stream is not raw, but encoded with MJPEG
