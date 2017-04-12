# RPi Meteor Station

Open source powered meteor station. We are currently using the Raspberry Pi 3 as the main development platform.
The software is still deep in the development phase, but here are its current features:

1. Video capturing from a sensitive B/W video camera
2. Compressing 256-frame blocks into the Four-frame Temporal Pixel (FTP) format
3. Detecting bright fireballs in real time
4. Detecting meteors on FTP compressed files
5. Extracting stars from FTP compressed files

You can see out Hackaday project web-page for more info: https://hackaday.io/project/6811-asteria-network


## Requirements
This guide will assume basic knowledge of electronics, the Unix environment, and some minor experience with the Raspberry Pi platform itself.

### Hardware
This code was designed to work with specific hardware, not because we wanted it so, but because only a unique combination of hardware will work at all. Thus, it is recommended that you follow the list as closely as possible.

1. **Raspberry Pi 2 or Raspberry Pi 3 single-board computer.**
The first version of the system was developed on the Raspberry Pi 2, while the system is now being tested on the RPi3, which is what we recommend you use, as it provides much more computing power. The system will NOT work with Raspberry Pi 1.

1. **Class 10 microSD card, 32GB or higher.** 
The recorded data takes up a lot of space, as much as several gigabytes per night. To be able to store at least one week of data on the system, a 64GB SD card is recommended.

1. **5V power supply for the RPi with the maximum current of at least 2A.** 
The RPi will have to power the video digitizer, and sometimes run at almost 100% CPU utilization, which draws a lot of power. We recommend using the official RPi 5V/2.5A power supply. Remember, most of the issues people have with RPis are caused by a power supply which is not powerful enough.

1. **RPi case with a fan + heatsinks.** 
If you end up using the RPi3, it is very probable that you will need to use a case with a fan, as the software will likely utilize the CPU close to 100% at some time. Be careful to buy a case with a fan which will not interfiere with the Real Time Clock module. We recommend buying a case which allows the fan to be mounted on the outside of the case.

1. **Real Time Clock module**
The RPi itself does not have a battery, so every time you turn it off, it loses the current time. The time then needs to be updates via the Internet. If for some reason you do not have access to the Internet, or you network connection is down, it is a good idea to have the correct time nevertheless, as it is essential for meteor trajectory estimation.
We recommend buying the DS3231 RTC module. See under [Guides/rpi3_rtc_setup.md](Guides/rpi3_rtc_setup.md) for information on installing this RTC module.

1. **EasyCap UTV007 video digitizer.** 
This device is used to digitize the video signal from the analogue camera, so it can be digitally processed and stored. We have tried numerous EasyCap devices, but **only the one the UTV007 chipset** works without any issues on the RPi.

1. **Sony Effio 673 CCTV camera and widefield lens (4mm or 6mm).**
This camera needs a 12V power supply. 

1. **Security camera housing.**
The best place to mount a meteor camera is on the outside wall of your house. As this means that the camera will be exposed to the elements, you need a good camera housing. We recommend that you get a housing with a heater and a fan, which will keep it warm in the winter and cool in the summer. Also, be sure to buy a housing large enough to accomodate your camera. There is one **important thing to note** - security camera housings are **not** designed to look up at the sky. Most of them are designed to be under a roof and looking down. As your camera will be looking up, and most likely be without the protection of a roof, you will have to properly isolate it. Buy some silirubber groutcone sealant and (after you fully assemble your camera and properly test everything), apply the sealant along all openings and joints, and most importantly, along the edges of the glass at the front. You want to keep the camera dry are prevent the humidity from getting inside. If you have some humidity inside the camera, when the temperature hits the dew point, everything inside the housing will be wet. People have also found that putting alumininum foil on the glass, from the inside of the housing, prevents the humidity from forming (just be careful not to obstruct the view of your camera).

1. **Wiring**
You will probably need some cables and connectors to connect your camera to the digitizer, and to bring power to you camera. We recommend using a shielded coaxial cable for the video signal, and a simple copper pair wire for the power (although you might want to get a shielded cable for power if there's a lot of interference in the video signal).

### Software
You will need the following software and libraries to run the code:

- Python 2.7
- OpenCV 2 for Python (2.4.9.1 or later)
- numpy (1.8.2 or later)
- scipy (0.18.1 or later)
- matplotlib (1.4.2 or later)
- cython (0.25.2 or later)
- pyephem (3.7.6.0 or later)

## Installation

### Enabling the watchdog service
A watchdog service is a service that occasionally checks if the RPi is responsive and if it's working fine. If the RPi hangs of freezes, it will reboot it. See under [Guides/enabling_watchdog.md](Guides/enabling_watchdog.md) for more information.

### Compiling the Kernel-based Hough Transform module
Navigate with terminal to base git directory, and run:

```
sudo python setup.py install
```

### Checking video device and initializing proper settings

