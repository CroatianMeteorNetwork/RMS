#!/bin/bash
# This software is part of the Linux port of RMS
# Copyright (C) 2023  Ed Harman
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This script will take an Ubuntu LTS or Debian-11/12 release and install all the components
# required to run an RMS meteor station.
# If you wish to run multiple stations on this host, after running this script execute:
# ~/source/RMS/Scripts/MultiCamlinux/add_GStation.sh

# Create source directory
mkdir -p ~/source
cd ~/source

# System updates
sudo apt-get update
sudo apt-get -y upgrade

# Install all required packages
sudo apt-get install -y \
    git wget zip openssh-server\
    python3 python3-dev python3-pip python3-tk python3-pil python3-gi python3-gst-1.0 \
    cmake mplayer \
    libblas-dev libatlas-base-dev liblapack-dev \
    libopencv-dev libffi-dev libssl-dev \
    libxml2-dev libxslt1-dev \
    libgirepository1.0-dev libcairo2-dev \
    at-spi2-core gir1.2-gstreamer-1.0 \
    socat chrony \
    imagemagick ffmpeg \
    qt5-qmake lxterminal \
    python3-virtualenv \
    gstreamer1.0-python3-plugin-loader \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gobject-introspection \
    gstreamer1.0-libav

# Configure system services
sudo timedatectl set-timezone UTC
sudo timedatectl set-local-rtc 0
sudo systemctl start chrony
sudo systemctl enable chrony
sudo systemctl start ssh
sudo systemctl enable ssh

# Clone repositories
git clone https://github.com/CroatianMeteorNetwork/RMS.git
git clone https://github.com/CroatianMeteorNetwork/cmn_binviewer.git

# Set up Python virtual environment
cd ~
virtualenv vRMS
source ~/vRMS/bin/activate

# Install Python packages
pip3 install --upgrade pip setuptools wheel
pip3 install -r ~/source/RMS/requirements.txt
pip3 install PyQt5

# Build and install OpenCV
cd ~/source/RMS
./opencv4_install.sh ~/vRMS

# Install RMS
python setup.py install

# Create desktop shortcuts if running in desktop environment
if [ -d "$HOME/Desktop" ]; then
    ~/source/RMS/Scripts/GenerateDesktopLinks.sh
fi