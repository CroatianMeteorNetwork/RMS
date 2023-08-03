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
# Version 1.1   - fixed bug introduced during testing
# Version 1.0	- initial release



#!/usr/bin/bash

echo -e "\n\n\nThis script automates the download and installation of RMS and configures support for multiple cameras on a generic Ubuntu-Desktop"
echo -e "\n If you wish to proceed type y or Y at the prompt or n/N to exit"
read -p  "Y/N: " Ans
case $Ans in 
        [nN]* ) 
        exit ;;
esac
unset Ans
mkdir ~/source
cd ~/source
sudo apt-get install -y git
git clone https://github.com/CroatianMeteorNetwork/RMS.git
source ~/source/RMS/Scripts/MultiCamLinux/RMSInstaller.sh
ln -s ~/source/RMS/Scripts/MultiCamLinux/icon.png ~/source/cmn_binviewer
echo -e "\n\n\nDo you wish to configure some stations?\n"
read -p  "Y/N: " Ans
case $Ans in 
        [nN]* ) 
        exit ;;
esac
cd ~/source/RMS/Scripts/MultiCamLinux
./add_GStation.sh

