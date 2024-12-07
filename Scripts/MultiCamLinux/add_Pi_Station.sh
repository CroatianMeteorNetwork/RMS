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
#
# Version 1.8	- replaced Desktop link with one that starts all captures
#
# Version 1.7   - changed desktop launcher display names and added conkyrc1 mod
#
# Version 1.6   - added support for RPi-5 platforms that can run max 4 cameras
#
# Version 1.5   - added support for non English locales where user user directories may not include a directory named Desktop
#		 i.e. this enables support of RMS on a non English distro install
#
# Version 1.4   - moved codebase into RMS/Scripts/MultiCamLinux
#
# Version 1.3	- fixed path to CMN desktop shortcut
#
# Version 1.2	- added a  change to the flag reboot_after_processing from true to false
#
# Version 1.1
# Changes	- added station arguments to  Launch scripts
#		- changed desktop links  for StartCapture to symbolic links of the scripts within .config/autostart
#
# Version 1.2	- Peter E. took over dev of this script and any blame going forward
#		- white space and indents added
#

MaxStation=4	# maximum number of stations allowed on a Pi5

if [[ $(uname -m ) == aarch64 ]]
then
    Model=$(lscpu| grep -o 'Cortex.*'|grep -o '[0-9]*')
    if [[ $Model -lt 76 ]]  # RPi5 uses Cortex A76  - use 72 for script testing on an RPi4
    then
	echo "This platform will not support multiple cameras, an RPi5 or similar is required"
	exit
    fi

    DefStation=$(awk ' /stationID:/ {print toupper($2)}' ~/source/RMS/.config) # force uppercase 
    if [[ $DefStation == XX0001  &&  ! -d ~/source/Stations/ ]]
    then
	echo "Please run RMS_Firstrun and configure your 1st station"
	echo "then if you wish to configure additional stations execute add_Pi_Station again"
	exit
    fi
    if [[ ! -d ~/source/Stations/ ]]
    then

cat <<EOF

Multiple cameras are supported by relocating the default camera configuration files -
- .config
- platepar_cmn2010.cal
- mask.bmp

from their default location  of ~/source/RMS -to a folder located at -

~/source/Stations/<StationID>

It appears you have already configured your first station - id ${DefStation}
so its config files will now be relocated to -
~/source/Stations/${DefStation}
Captured data from your default station are stored by default in

~/RMS_data

This data will be moved to -
~/RMS_data/${DefStation}
EOF

	# First station has been configured, so move it to ~/source/Stations/<station-ID>, 
	#  and move any captured data to ~/RMS_data/<station-ID>. 

	# tweak conky station title
	sed -i 's/\(source\)\/RMS/\1\/Stations\/'"$DefStation"'/g' /home/rms/.conkyrc1
	# tweak conky data source path for log data
	sed -i "s/\(.*RMS_data\)/\1\/${DefStation}/g" /home/rms/.conkyrc1

	mkdir ~/source/Stations/
	mkdir ~/source/Stations/${DefStation}
	cp ~/source/RMS/.config ~/source/Stations/${DefStation}
       	cd ~/source/RMS
	sed -i "s,data_dir.*$,data_dir: ~/RMS_data/${DefStation},g" ~/source/Stations/${DefStation}/.config
	if [[ -e ~/source/RMS/platepar_cmn2010.cal ]]
	then
	    mv  ~/source/RMS/platepar_cmn2010.cal ~/source/Stations/${DefStation}/
	fi
	if  [[ -e ~/source/RMS/mask.bmp ]]
	then
	    mv ~/source/RMS/mask.bmp ~/source/Stations/${DefStation}/
	#replace with copy from master
	cd ~/source/RMS
	wget -q https://raw.githubusercontent.com/CroatianMeteorNetwork/RMS/master/mask.bmp
	fi
	mkdir ~/RMS_data/${DefStation}
	cd  ~/RMS_data/${DefStation}
	mv ../ArchivedFiles .
	mv ../CapturedFiles .
	mv ../logs .

	cat <<- EOF > ~/Desktop/${DefStation}_StartCapture.desktop
	[Desktop Entry]
	Name=${DefStation}-StartCapture
	Type=Application
	Exec=lxterminal --title=${DefStation} -e "~/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${DefStation}"
	Hidden=false
	NoDisplay=false
	Icon=lxterminal
	EOF
	    chmod +x ~/Desktop/${DefStation}_StartCapture.desktop

	cat <<- EOF > ~/Desktop/${DefStation}-Show_LiveStream.desktop
	[Desktop Entry]
	Name=${DefStation}-ShowLiveStream
	Type=Application
	Exec=lxterminal --title=${DefStation}-LiveStream -e "~/source/RMS/Scripts/MultiCamLinux/LiveStream.sh ${DefStation}"
	Hidden=false
	NoDisplay=false
	Icon=lxterminal
	EOF
	chmod +x ~/Desktop/${DefStation}-Show_LiveStream.desktop

	# Check if the user's desktop directory environment variable is set
	if [ -n "$xdg-user-dir DESKTOP" ]
	then
	   Desktop=`xdg-user-dir DESKTOP`
	fi

	# cleanup erroneous Desktop shortcuts 
	if [[ -f ~/Desktop/RMS_FirstRun.sh ]]
	then
	    rm  ~/Desktop/CMNbinViewer.sh ~/Desktop/RMS_ShowLiveStream.sh ~/Desktop/RMS_StartCapture.sh
	    rm  ~/Desktop/RMS_config.txt ~/Desktop/TunnelIPCamera.sh ~/Desktop/DownloadOpenVPNconfig.sh
	fi
	
	# remove comment from last line of wayfire.ini to enable window cascade 
	sed -i s/#mode/mode/ ~/.config/wayfire.ini

    fi	# if [[ ! -d ~/source/Stations/ ]]

fi    # if [[ $(uname -m ) == aarch64 ]]

# need to count the number of directories under ~/source/Stations and only 
# prompt for adding the correct number of new cameras, so subtract that number
# from MaxStation defined above

numdirs=$(find ~/source/Stations/ -maxdepth 1 -type d | wc -l)
numdirs=$((numdirs-1))
Remaining=$((MaxStation-numdirs))
#echo MaxStation:$MaxStation NumDirs:$numdirs  Remaining:$Remaining

declare Station
while :
do
	if [[ ${#Station[@]} ==  $Remaining ]]
	then
	    echo ""
	    echo "Done, This platform has a maximum of $numdirs cameras"
	    break
	fi
	read -p "Enter station ID, <cr> to end: " this_Station
	this_Station="${this_Station^^}"   #uppercase it..
	if [[ -z $this_Station ]]
	then
	    break
	fi
	Station+=("$this_Station")
	# echo $Station
done

echo -e "\nNew stations to add -"
printf '%s\n' "${Station[@]}"

# check if ~/${RMS_data} exists and if not create it....
if [[ ! -d "${RMS_data}/" ]]
then
    mkdir ${RMS_data}
fi

No_Stations=${#Station[@]}
for item in "${Station[@]}"
do
	if [[ -d ~/source/Stations/${item} ]]
	then
	    echo -e "\n\nNot creating station ${item} - it already exists\n"
	    exit
	else
	    echo "making dir Stations/${item}"
	    mkdir ~/source/Stations/${item} 
 	    if [[ ! -d ${RMS_data}/$item ]]
	    then
		mkdir ~/RMS_data/${item}
	    fi
	fi

	cp  ~/source/RMS/.config ~/source/Stations/${item}

	# create autostart entry for the station
	cat <<- EOF > ~/Desktop/${item}_StartCapture.desktop
	[Desktop Entry]
	Name=${item}-StartCapture
	Type=Application
	Exec=lxterminal --title=${item} -e "~/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${item}"
	Hidden=false
	NoDisplay=false
	Icon=lxterminal
	EOF
	chmod +x ~/Desktop/${item}_StartCapture.desktop

	if  [[ -e ~/source/RMS/mask.bmp ]]
	then
    	    cp ~/source/RMS/mask.bmp ~/source/Stations/${item}/
	fi

	# create a ShowLiveStream-<station and a desktop shortcut
	cat <<- EOF > ~/Desktop/${item}-Show_LiveStream.desktop
	[Desktop Entry]
	Name=${item}-ShowLiveStream
	Type=Application
	Exec=lxterminal --title=Stream-${item} -e "~/source/RMS/Scripts/MultiCamLinux/LiveStream.sh ${item}"
	Hidden=false
	NoDisplay=false
	Icon=lxterminal
	EOF
	chmod +x ~/Desktop/${item}-Show_LiveStream.desktop

	# customise each .config
	# set the station_id
	sed -i  "s/D:.*$/D: $item/g" ~/source/Stations/${item}/.config 
	# set the ${RMS_data} dir
	sed -i "s,data_dir.*$,data_dir: ~/RMS_data/${item},g" ~/source/Stations/${item}/.config
	echo -e "\n\nAdded station $item\n\n"
	# disable daily post processing reboot
	sed -i "s/\(reboot_after_processing:\).*/\1 false/g" ~/source/Stations/${item}/.config
done

# check if keys are  already present
if [[ -f ~/.ssh/id_rsa ]]
then
    if [ ! -z ${Configured} ]
    then
	echo "SSH keys already exist, not overwriting them"
    fi
else
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -P ""
    cp ~/.ssh/id_rsa.pub ~/Desktop
    echo "SSH keys successfully generated in ~/.ssh"
    echo "your new id_rsa.pub public key file now placed on the Desktop"
    echo "Be sure to send a copy of this file to Denis"
fi

if [[ ! -v Model ]]
then
    timedatectl set-timezone UTC
fi

echo -e "\n\nStation configuration complete\n\n"
user=`id -u -n`

# Set up shortcut for CMNbinViewer
cat <<- EOF > ~/Desktop/CMNbinViewer.desktop
[Desktop Entry]
Name=CMNbinViewer
Type=Application
Exec=/home/${user}/source/RMS/Scripts/CMNbinViewer_env.sh
Hidden=false
NoDisplay=false
Icon=/home/${user}/source/RMS/Scripts/MultiCamLinux/icon.png
EOF
chmod +x ~/Desktop/CMNbinViewer.desktop

# set the option to allow launching of desktop shortcuts
#sed -i s/quick_exec=0/quick_exec=1/ ~/.config/libfm/libfm.conf

if grep -Fqv "UTC" /etc/timezone
then
    sudo timedatectl set-timezone UTC
fi

# set the extra_space for all configured stations -
mult=$(ls -1 /home/${USER}/source/Stations/|wc -l)
for Dir in /home/${USER}/source/Stations/*
    do
	sed -i "s/extra_space_gb: .*$/extra_space_gb: $(( ${mult} * 30 ))/g" ${Dir}/.config
done

# set up so RMS_FirstRun can autostart all captures
rm -f /home/${USER}/Desktop/RMS_StartCapture.sh
ln -s /home/${USER}/source/RMS/Scripts/MultiCamLinux/Pi/RMS_StartCapture_MCP.sh /home/${USER}/Desktop/RMS_StartCapture.sh

# get a copy of the default .config
cd ~/source/RMS
rm .config  #delete potentially modified .config and replace with new..
wget -q https://raw.githubusercontent.com/CroatianMeteorNetwork/RMS/master/.config
