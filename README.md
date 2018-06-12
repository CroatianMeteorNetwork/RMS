# RPi Meteor Station

Open source powered meteor station. We are currently using the Raspberry Pi 3 as the main development platform. **The code also works on Linux PCs.**
The software is still in the development phase, but here are the current features:

1. Automated video capture - start at dusk, stop at dawn. Analog cameras supported through EasyCap, **IP cameras up to 720p resolution - CONTACT US FOR MORE DETAILS.**
1. Compressing 256-frame blocks into the Four-frame Temporal Pixel (FTP) format (see [Jenniskens et al., 2011 CAMS](http://cams.seti.org/CAMSoverviewpaper.pdf) paper for more info).
1. Detecting bright fireballs in real time
1. Detecting meteors on FTP compressed files
1. Extracting stars from FTP compressed files
1. Astrometry and photometry calibration
1. Automatic recalibration of astrometry every night
1. Automatic upload of calibrated detections to central server
1. Manual reduction of fireballs/meteors

You can see out Hackaday project web-page for more info: https://hackaday.io/project/6811-asteria-network


## Requirements
This guide will assume basic knowledge of electronics, the Unix environment, and some minor experience with the Raspberry Pi platform itself.

### Hardware
This code was designed to work with specific hardware, not because we wanted it so, but because only a unique combination of hardware will work at all. Thus, it is recommended that you follow the list as closely as possible. 
You can find a **step-by-step guide how to assemble the hardware and install the software on Instructables** (NOTE: we have an SD card image available too, see below for more details): http://www.instructables.com/id/Raspberry-Pi-Meteor-Station/

#### RPi

1. **Raspberry Pi 3 single-board computer.**
The first version of the system was developed on the Raspberry Pi 2, while the system is now being tested on the RPi3, which is what we recommend you use, as it provides much more computing power. The code will NOT work on Raspberry Pi 1.

1. **Class 10 microSD card, 64GB or higher.** 
The recorded data takes up a lot of space, as much as several gigabytes per night. To be able to store at least one week of data on the system, a 64GB SD card is the minimum.

1. **5V power supply for the RPi with the maximum current of at least 2.5A.** 
The RPi will have to power the video digitizer, and sometimes run at almost 100% CPU utilization, which draws a lot of power. We recommend using the official RPi 5V/2.5A power supply. Remember, most of the issues people have with RPis are caused by a power supply which is not powerful enough.

1. **RPi case with a fan + heatsinks.** 
You will need to use a RPi case with a fan, as the software will likely utilize the CPU close to 100% at some time. Be careful to buy a case with a fan which will not interfere with the Real Time Clock module. We recommend buying a case which allows the fan to be mounted on the outside of the case.

1. **Real Time Clock module**
The RPi itself does not have a battery, so every time you turn it off, it loses the current time. The time then needs to be fetched from the Internet. If for some reason you do not have access to the Internet, or you network connection is down, it is a good idea to have the correct time nevertheless, as it is essential for meteor trajectory estimation.
We recommend buying the DS3231 RTC module. See under [Guides/rpi3_rtc_setup.md](Guides/rpi3_rtc_setup.md) for information on installing this RTC module.

#### Cameras

##### Analog

1. **EasyCap UTV007 video digitizer.** 
This device is used to digitize the video signal from the analogue camera, so it can be digitally processed and stored. We have tried numerous EasyCap devices, but **only the one the UTV007 chipset** works without any issues on the RPi.

1. **Sony Effio 673 CCTV camera and widefield lens (4mm or 6mm).**
Upon thorough testing, it was found that this camera has the best price–performance ratio, especially when it is paired with a wide-field lens. A 4mm wide field lens will give a field-of-view of about 64x48 degrees. This camera needs a 12V power supply. One important thing to note is that the camera needs to have the IR-cut filter removed (this filter will filter out all infrared light, but as meteors are radiating in that part of the spectrum, we want to record that light as well). Alternatives to the proposed camera are possible, see [Samuels et al. (2014) "Performance of 
new low-cost 1/3" security cameras for meteor surveillance"](http://www.imo.net/imcs/imc2014/2014-29-wray-final.pdf) for more information.

---------

OR:

##### Digital IP


1. **IP camera**
Preferably an IMX225 or IMX291 based camera. This part is still in the testing phase, but contact us for more details!

---------

1. **Security camera housing.**
The best place to mount a meteor camera is on the outside wall of your house. As this means that the camera will be exposed to the elements, you need a good camera housing. We recommend that you get a housing with a heater and a fan, which will keep it warm in the winter and cool in the summer. Also, be sure to buy a housing large enough to accomodate your camera. There is one **important thing to note** - security camera housings are **not** designed to look up at the sky. Most of them are designed to be under a roof and looking down. As your camera will be looking up, and most likely be without the protection of a roof, you will have to properly isolate it. Buy some silirubber groutcone sealant and (after you fully assemble your camera and properly test everything), apply the sealant along all openings and joints, and most importantly, along the edges of the glass at the front. You want to keep the camera dry are prevent the humidity from getting inside. If you have some humidity inside the camera, when the temperature hits the dew point, everything inside the housing will be wet. People have also found that putting alumininum foil on the glass, from the inside of the housing, prevents the humidity from forming (just be careful not to obstruct the view of your camera).

1. **Wiring**
You will probably need some cables and connectors to connect your camera to the digitizer, and to bring power to you camera. We recommend using a shielded coaxial cable for the video signal, and a simple copper pair wire for the power (although you might want to get a shielded cable for power if there's a lot of interference in the video signal).



### Software

**NOTE:** We have an SD card image with everything installed on it. We don't want to distribute it publically just yet as it
s not 100% tested, but contact us if you want a copy and more details. Then you'll just have to flash it to an SD card and that's it!

---------

Set up your Raspberry Pi with Raspbian Jessie operating system (gstreamer does not really work on Stretch, and it's necessary if you want to run an IP camera). Here's the guide which explaing how to do just that: [Installing Raspbian](https://www.raspberrypi.org/documentation/installation/installing-images/)

Furthermore, you will need the following software and libraries to run the code:

- git
- mplayer
- Python2.7 or Python 3.5
- python2.7-dev or python3-dev
- libblas-dev liblapack-dev
- libffi-dev libssl-dev
- Python libraries:
	- gitpython
	- astropy
	- OpenCV 3 for Python
	- PIL (i.e. python-imaging-tk)
	- numpy (1.14.0 or later)
	- scipy (1.0.0 or later)
	- matplotlib (2.0.0 or later)
	- cython (0.25.2 or later)
	- pyephem (3.7.6.0 or later)
	- paramiko
	
All python libraries will be installed when you run the setup.py script (instructions below). If you want use IP cameras, you need to install a specail compilation of OpenCV that supports gstreamer. Run the opencv3_install.sh scripts that is provided with the code.

## Setting up

### Setting the timezone to UTC
It is always a good idea to set the timezone to UTC when recording any data. This provides a common time reference among observatons, and more than once there have been issues when people were using different time zones. So, use your favorite search engine to find how to change the timezone on your RPi to UTC.


### Enabling the watchdog service
A watchdog service is a service that occasionally checks if the RPi is responsive and if it's working fine. If the RPi hangs of freezes, it will reboot it. See under [Guides/enabling_watchdog.md](Guides/enabling_watchdog.md) for more information.


### Getting this code
First, find directory where you want to download the code. If you don't care, I presume the home directory /home/pi is fine.
The simplest way of obtaining this code is by opening the terminal and running:

```
git clone https://github.com/CroatianMeteorNetwork/RMS.git
```

This will download the code in this repository in the RMS directory. 


### Running setup.py and compiling the Kernel-based Hough Transform module
Navigate with terminal to base git directory (e.g. /home/pi/RMS/), and run:

```
sudo python setup.py install
```

This will compile the code in C++ which we are using as one of the processing steps in meteor detection. The method in question is called Kernel-based Hough Transform, and you can read more about it here: [KHT](http://www2.ic.uff.br/~laffernandes/projects/kht/)

This will also install all Python libraries that you might need, except OpenCV. To install OpenCV, open the terminal and run:

```
sudo apt-get install libopencv-dev python-opencv
```

### Checking video device and initializing proper settings - ANALOG CAMERAS ONLY!
Once you connect the EasyCap digitizer and the camera, you need to check if the video signal is being properly received.

#### NTSC
If you have a NTSC camera (i.e. you are living in North America), run this in the terminal:

```
mplayer tv:// -tv driver=v4l2:device=/dev/video0:input=0:norm=NTSC
```

#### PAL

If you are in Europe, you most likely have a PAL camera, not NTSC. There is a 'hack' you can use to force the EasyCap UTV007 to set itself up in the PAL format. Run this in the terminal:

```
mplayer tv:// -vo null
```
After a few seconds, kill the script with Ctrl+C. Now, you can see the video if you run:

```
mplayer tv:// -tv driver=v4l2:device=/dev/video0:input=0:norm=PAL
```

### Editing the configuration file
This is a very important step as all settings are read from the configuration file. The file in question is the [.config](.config) file. Once you download this repository, start editing the file with your favorite editor.

#### [System]
##### Station ID
This is currenlty arbitrary - as long as it's a 3 digit number, you are fine.

##### GPS location
Edit the latitude, longitude and elevation of the location of your camera. This is used for automatically calculating the starting and ending time of the time of capture, as well as the astrometric plates. Try to be as precise as possible, possibly measuring the location of the camera with the GPS on your phone.


#### [Capture]
#### Resolution and FPS
To be able to capture the video properly, you need to set up the right resolution and FPS (frames per second). Here is a table giving the numbers for the two video standards.

| Option | PAL | NTSC |
|--------|-----|------|
| width  |720  |720   |
| height |576  |480   |
| fps    |25.0 |29.97 |




## Running the code

### Capturing video and saving data
To start the video capture, navigate to the base folder (e.g. /home/pi/RMS) with the terminal and run:

```
python -m RMS.StartCapture
```

This command will automatically start capturing upon sunset, and stop capturing upon sunrise.

If you want to start capture right away, for a specified duration, run this command with the argument -d HH.hh, where HH.hh is the number of hours you want to run the capture for. E.g. if you want to capture the video for one and a half hours, you would run:

```
python -m RMS.StartCapture -d 1.5
```

The data will be saved in /home/pi/RMS_data/YYYYMMDD_hhmmss_uuuuuu, where YYYYMMDD_hhmmss_uuuuuu is the timestamp of the time when the recording was started, which is used as a name for the directory where the data for the night will be stored. 

#### Viewing FF bin files (compressed video data)
You can view the recorded data using the [CMN_binViewer](https://github.com/CroatianMeteorNetwork/cmn_binviewer) software.

#### Viewing FR bin files (fireball detections)
You may notice that there are some FR files in the night directory (as opposed to FF, which are storing the compressed video data). The FR files are created by the fireball detector, which detects brighter meteors as well. To see the detection, run:

```
python -m Utils.FRbinViewer ~/RMS_data/YYYMMDD_hhmmss_uuuuuu
```
where YYYMMDD_hhmmss_uuuuuu is the name of the night directory.

### Star extraction and meteor detection
To extract the stars on recorded images, and detect meteors on them, run:

```
python -m RMS.DetectStarsAndMeteors ~/RMS_data/YYYMMDD_hhmmss_uuuuuu
```

where YYYMMDD_hhmmss_uuuuuu is the name of the night directory. This will take a while to run, and when it is done, it will generate a file called CALSTARS (which will hold information about detected stars on images), and FTPdetectinfo file which will hold information about detected meteors.


## Donate!

This project is run by volounteers in their free time. Show you appreciation for the project by donating!

**Crypo wallets:**

- ***Bitcoin:*** 3M3toMCSxzEvxtLHmR63LBfGeYbruAWvF1
- ***Bitcoin Cash:*** 1D7NbwZ7p61sgXA5kqUNAUHHyvGkfJiP9s
- ***Bitcoin Gold:*** GbfbNuAn58DKYNvEWMf9m2Fbcupc8xoPJj
- ***Ethereum:*** 0x2731cbd765514c6a2675be46bfff05bd4b8a5945
- ***Litecoin:*** LiBZkdgzEooJNfrmcYw4MQtUK8ZBGYmuix


## Citations

For academic use, please cite the paper:
>Vida, D., Zubović, D., Šegon, D., Gural, P., & Cupec, R. (2016). *Open-source meteor detection software for low-cost single-board computers*. **Proceedings of the IMC2016, Egmond, The Netherlands**, pp. 307-318