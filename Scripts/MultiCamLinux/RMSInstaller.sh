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
#!/bin/bash
#
# This is script will take an Ubuntu LTS or Debian-11/12 release and install all the components
# required to run an RMS meteor station
# If you wish to run multiple stations on this host, after running this script execute  -
# ~/source/RMS/Scripts/MultiCamlinux/add_GStation.sh
#

cd ~/
mkdir source
cd  source
sudo apt install -y git wget zip
git clone https://github.com/CroatianMeteorNetwork/RMS.git
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-tk libxslt1-dev python3-pil
sudo apt-get install -y git mplayer python3 python3-dev python3-pip libblas-dev libatlas-base-dev \
liblapack-dev at-spi2-core libopencv-dev libffi-dev libssl-dev socat ntp \
libxml2-dev libxslt-dev imagemagick ffmpeg cmake 
sudo apt install -y python3-gi python3-gst-1.0 libgirepository1.0-dev libcairo2-dev gir1.2-gstreamer-1.0

pip3 install --upgrade pip
if [[ $(awk '{print $3}' /etc/issue) == 11 ]]
    then
    sudo apt install virtualenv     # Debian <12 package this seperately
fi
sudo apt install -y python3-virtualenv
cd ~
virtualenv vRMS
source ~/vRMS/bin/activate
pip3 install -U pip
pip install -r ~/source/RMS/requirements.txt
pip install tflite-runtime    # missed from requirements due to python 3.11
pip install PyQt5
pip install pyqtgraph
pip install pycairo
pip install PyGObject
cd ~/source/RMS
#sudo apt install -y gstreamer1.0*  # fails in certain env's, manually install good, bad and libavcodec-dev
sudo apt install -y gstreamer1.0-python3-dbg-plugin-loader
sudo apt install -y gstreamer1.0-python3-plugin-loader
sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Check what platform we are on, x86_64 platforms don't support NEON extensions that are ARM specific
if [[ $(uname -m) == x86_64 ]]
    then
    ex +g/NEON/d -cwq opencv4_install.sh
fi

./opencv4_install.sh ~/vRMS
cd ~/source/RMS
python setup.py install
sudo apt install -y gstreamer1.0-plugins-good
# get CMNbinViewer....
cd ~/source
git clone https://github.com/CroatianMeteorNetwork/cmn_binviewer.git
# check to see if a desktop is installed - not foolproof - doesnt check for X11 env etc..
if [ -d "$HOME/Desktop" ]
then 
# generate desktop links
~/source/RMS/Scripts/GenerateDesktopLinks.sh
fi
