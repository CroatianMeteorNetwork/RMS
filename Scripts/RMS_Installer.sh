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
    git wget zip openssh-server \
    python3 python3-dev python3-pip python3-tk python3-pil \
    mplayer \
    socat chrony \
    imagemagick ffmpeg \
    lxterminal \
    python3-venv \
    python3-gi python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-opencv \
    python3-pyqt5

# Install gstreamer python plugin loader if available (not present on all platforms)
sudo apt-get install -y gstreamer1.0-python3-plugin-loader 2>/dev/null || true

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

# Set up Python virtual environment with access to system packages
cd ~
python3 -m venv --system-site-packages vRMS
source ~/vRMS/bin/activate

# Install Python packages
python -m pip install cython
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r ~/source/RMS/requirements.txt

# Install RMS
cd ~/source/RMS
python -m pip install -e . --no-deps --no-build-isolation

# Create desktop shortcuts if running in desktop environment
if [ -d "$HOME/Desktop" ]; then
    ~/source/RMS/Scripts/GenerateDesktopLinks.sh
fi
