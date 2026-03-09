# Raspberry Pi Meteor Station (RMS)

RMS is the core open-source software library powering the **[Global Meteor Network (GMN)](https://globalmeteornetwork.org/)**. 

The GMN is a worldwide, collaborative network of amateur and professional astronomers operating low-cost meteor cameras. Its primary goal is to observe meteors in the night sky, compute their trajectories and orbits, and monitor the near-Earth meteoroid environment. 

The RMS software provides a complete, automated pipeline for meteor observation. It currently uses digital IP cameras and is designed primarily for the Raspberry Pi platform (Pi 4 and Pi 5 recommended), though it also fully supports Linux PCs. 

### Current Features
1. **Automated video capture:** Automatically starts recording at dusk and stops at dawn.
2. **Efficient data storage:** Compresses 256-frame video blocks into the Four-frame Temporal Pixel (FTP) format (see [Jenniskens et al., 2011 CAMS](http://cams.seti.org/CAMSoverviewpaper.pdf)).
3. **Real-time detection:** Detects bright fireballs as they happen.
4. **Meteor processing:** Detects meteors and extracts stars from FTP compressed files.
5. **Calibration:** Astrometric and photometric calibration, with automatic nightly recalibration.
6. **Data syncing:** Automatic upload of calibrated detections to the central server.
7. **Manual reduction:** Tools for the manual reduction of fireballs and meteors.

For more information on the project, required hardware, and how to get involved, please visit the **[GMN Wiki Main Page](https://globalmeteornetwork.org/wiki/index.php?title=Main_Page)**.

---

## Table of Contents
1. [Installation](#installation)
   - [Raspberry Pi](#raspberry-pi)
   - [Linux Systems](#linux-systems)
2. [Custom/Developer Installation (Conda)](#customdeveloper-installation-conda)
3. [Running the Code](#running-the-code)
4. [Citations](#citations)

---

## Installation

We highly recommend using the pre-built images or automated installation scripts provided on our Wiki. This is by far the easiest and most stable way to get your meteor station up and running.

### Raspberry Pi
If you are building a standard GMN camera using a Raspberry Pi, please follow the comprehensive step-by-step guides on our Wiki:
* **[Build, Install, & Setup Your Camera - The Complete How-To](https://globalmeteornetwork.org/wiki/index.php?title=Build_%26_Install_%26_Setup_your_camera_-_The_complete_how-to)**
* **[Installing OS onto a Raspberry Pi](https://globalmeteornetwork.org/wiki/index.php?title=Installing_OS_onto_a_Raspberry_Pi)** (Direct link to flashing the RMS image)

### Linux Systems
If you want to run RMS natively on a dedicated Linux PC or configure a multi-camera setup, please refer to the dedicated Linux installation guide on the Wiki:
* **[Advanced RMS Installations and Multi-Camera Support](https://globalmeteornetwork.org/wiki/index.php?title=Advanced_RMS_installations_and_Multi-camera_support)**

*(Note: Meteor detection does not run natively on Windows due to multiprocessing limitations, but you can still run submodules like astrometric calibration or manual reduction.)*

---

## Custom/Developer Installation (Conda)

**NOTE: These instructions are intended ONLY for advanced users, developers, or highly custom installations. Standard users should use the Wiki links above.**

If you are using Anaconda Python on your Windows or Linux PC and want to install the libraries manually, run the following commands in your Anaconda environment:

```bash
conda create -y -n rms -c conda-forge python==3.11.6
conda activate rms
conda install -y -c conda-forge numpy'<2.0' scipy gitpython cython matplotlib paramiko
conda install -y -c conda-forge numba
conda install -y pyqt==5.15.10
conda install -y -c conda-forge Pillow pyqtgraph==0.12.3
conda install -y -c conda-forge ephem
conda install -y -c conda-forge imageio
conda install -y -c conda-forge pygobject
conda install -y -c conda-forge opencv
conda install -y -c astropy astropy
pip install rawpy'<0.22'
pip install git+[https://github.com/matejak/imreg_dft@master#egg=imreg_dft](https://github.com/matejak/imreg_dft@master#egg=imreg_dft)'>2.0.0'
pip install astrometry
```

If you want full gstreamer support (for better capture and raw video recording), you need to install the additional gstreamer libraries:

```bash
conda install -y -c conda-forge gstreamer==1.22.3 gobject-introspection gst-libav gst-plugins-bad gst-plugins-base gst-plugins-good gst-plugins-ugly
```

Next, clone this repository and install the RMS code as a package:

```bash
git clone [https://github.com/CroatianMeteorNetwork/RMS.git](https://github.com/CroatianMeteorNetwork/RMS.git)
cd RMS
pip install .
```

---

## Running the Code

Once your system is fully set up and configured (via the `.config` file, as explained in the Wiki), you can run various RMS modules. 

### Capturing video and saving data
To start the automated video capture, navigate to your base RMS folder in the terminal and run:

```bash
python -m RMS.StartCapture
```

This command will automatically start capturing upon sunset, stop at sunrise, perform the detection automatically, do the astrometric recalibration, and upload the detections to the server.

If you want to start the capture right away for a specified duration, run this command with the `-d` argument (e.g., for 1.5 hours):
```bash
python -m RMS.StartCapture -d 1.5
```

### Live Stream
To test your camera with the RMS configuration without running a real capture session, you can run the Live Stream module:
```bash
python -m Utils.ShowLiveStream
```

### Viewing FF bin files (compressed video data)
You can view the recorded FTP data using the [CMN_binViewer](https://github.com/CroatianMeteorNetwork/cmn_binviewer) software. 

### Viewing FR bin files (fireball detections)
The fireball detector generates `.FR` files in the night directory for brighter meteors. To see the detection, run:
```bash
python -m Utils.FRbinViewer ~/RMS_data/YYYYMMDD_hhmmss_uuuuuu
```
*(Replace `YYYYMMDD_hhmmss_uuuuuu` with the exact name of the night directory).*

### Manual Star Extraction and Meteor Detection
This step is done automatically when `StartCapture` is run, but if you want to redo the detection manually on recorded images, run:
```bash
python -m RMS.DetectStarsAndMeteors ~/RMS_data/YYYYMMDD_hhmmss_uuuuuu
```
This generates a `CALSTARS` file (information about detected stars) and an `FTPdetectinfo` file (information about detected meteors).

---

## Citations

For academic use, please cite the paper:
> [Vida, D., Šegon, D., Gural, P.S., Brown, P.G., McIntyre, M.J., Dijkema, T.J., Pavletić, L., Kukić, P., Mazur, M.J., Eschman, P. and Roggemans, P., 2021. The Global Meteor Network–Methodology and first results. Monthly Notices of the Royal Astronomical Society, 506(4), pp.5046-5074.](https://academic.oup.com/mnras/article/506/4/5046/6347233)