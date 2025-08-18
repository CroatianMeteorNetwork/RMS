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
# Version 1.6   - added logic to prevent running this script on a Raspberry Pi
#                 added by Peter E., June 2024
#
# Version 1.5   - added support for non English locales where user user directories may not include a directory named Desktop
#                 i.e. this enables support of RMS on a non English distro install
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
# 
# Prevent running this script on a Raspberry Pi, because 
#  add_Pi_Station.sh should be used to add cameras on a Pi.
file=/sys/firmware/devicetree/base/model
if [[ -f "$file" ]]; then
    echo ""
    contents=$(tr -d '\0' < $file)
    echo "The file $file reads: $contents"
    if [[ $contents == 'Raspberry'* ]]; then
	echo "The add_GStation.sh script should not be used on Raspberry Pi."
	echo "Please use add_Pi_Station.sh to add cameras on a Pi5."
	echo ""
	exit 1
    fi
fi

# Check if the user's desktop directory environment variable is set
if [ -n "$xdg-user-dir DESKTOP" ]; then
    Desktop=`xdg-user-dir DESKTOP`
fi
#
if [[ "$#" -eq 0 ]]	#called with no arg
then
	RMS_data=~/RMS_data
	else
	RMS_data=${1}
fi
# cleanup erroneous Desktop shortcuts 
if [[ -f ~/Desktop/RMS_FirstRun.sh ]]
	then
	rm ~/Desktop/RMS_FirstRun.sh ~/Desktop/RMS_StartCapture.sh ~/Desktop/RMS_ShowLiveStream.sh ~/Desktop/TunnelIPCamera.sh ~/Desktop/CMNbinViewer.sh
fi

# Install the Buster RMS default terminal - lxterminal
if ! command -v lxterminal &> /dev/null 
then
	sudo apt-get install -y lxterminal &
	wait %1 
	echo -e "\n\nlaunching lxterminal\n\n"
# we need to launch lxterminal at least once for it to write its default preferences file
	echo -e "\n\nclosed lxterminal\n\n"
	# now lxterminal is installed and written it's config we can customise it
	sed -i "s/scrollback=.*/scrollback=10000/g" ~/.config/lxterminal/lxterminal.conf
	sed -i "s/geometry_columns=.*/geometry_columns=120/g" ~/.config/lxterminal/lxterminal.conf
	sed -i "s/geometry_rows=.*/geometry_rows=25/g" ~/.config/lxterminal/lxterminal.conf
	echo -e "\n\nlxterminal installed and preferences set\n\n"
fi
if [[ ! -d ~/source/Stations ]]
	then
	mkdir ~/source/Stations
fi 
declare Station
while :
do
        read -p "Enter station ID, <cr> to end: " this_Station
        if [[ -z $this_Station ]]
        then
                break
        fi
        Station+=("$this_Station")
#       echo $Station
done

echo -e "\nNew stations to add -"
printf '%s\n' "${Station[@]}"

# check if ~/${RMS_data} exists and if not create it....
if [[ ! -d "${RMS_data}/" ]]
        then
        mkdir ${RMS_data}
fi

# check if user has an autostart dir in their .config directory and if not create it...
if [[ ! -d ~/.config/autostart ]]
	then
	mkdir ~/.config/autostart
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
                mkdir ${RMS_data}/${item}
			fi

		fi
    cp  ~/source/RMS/.config ~/source/Stations/${item}
	# create autostart entry for the station
	cat <<- EOF > ~/.config/autostart/${item}_StartCap.desktop
	[Desktop Entry]
	Name=${item}-Startcapture
	Type=Application
	Exec=lxterminal --title=${item} -e "~/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${item}"
	Hidden=false
	NoDisplay=false
	Icon=lxterminal
	EOF
	# create Desktop softlink for StartCapture
	chmod +x ~/.config/autostart/${item}_StartCap.desktop
	ln -s ~/.config/autostart/${item}_StartCap.desktop ~/Desktop/${item}_StartCap.desktop
	chmod -x ~/Desktop/${item}_StartCap.desktop
	gio set ~/Desktop/${item}_StartCap.desktop metadata::trusted true
    chmod +x ~/Desktop/${item}_StartCap.desktop
    cp  ~/source/RMS/mask.bmp ~/source/Stations/${item}
	# create a ShowLiveStream-<station and a desktop shortcut 
	touch ~/Desktop/Show_LiveStream-${item}.desktop
	cat <<- EOF > ~/Desktop/Show_LiveStream-${item}.desktop
	[Desktop Entry]
	Name=${item}-ShowLiveStream
	Type=Application
	Exec=lxterminal --title=Stream-${item} -e "~/source/RMS/Scripts/MultiCamLinux/LiveStream.sh ${item}"
	Hidden=false
	NoDisplay=false
	Icon=lxterminal
	EOF
	chmod -x ~/Desktop/Show_LiveStream-${item}.desktop
	gio set ~/Desktop/Show_LiveStream-${item}.desktop metadata::trusted true
    chmod +x ~/Desktop/Show_LiveStream-${item}.desktop
	#
	# 			customise each .config
	#
	# set the station_id
	sed -i  "s/D:.*$/D: $item/g" ~/source/Stations/${item}/.config 
	# set the ${RMS_data} dir
	sed -i "s,data_dir.*$,data_dir: ${RMS_data}/${item},g" ~/source/Stations/${item}/.config
	# update the free space in ratio of number of stations being added, using a 20Gb factor for now
	sed -i "s/extra_space_gb: .*$/extra_space_gb: $(( ${No_Stations} * 20 ))/g" ~/source/Stations/${item}/.config
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
        echo "SSH keys successfully generated in ~/.ssh"
fi
timedatectl set-timezone UTC
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'    # disables inactivity lock
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0     # sleep inactivity 0=='never'
gsettings set org.gnome.desktop.session idle-delay 600   # screen blank after xxx secs, adjust to taste 0=='never'
gsettings set org.gnome.desktop.screensaver lock-enabled false		# disable screensaver lock

# only prompt for these optional installs if the user has run this for the 1st time

#  launch the vRMS environment at logon
if ( ! grep -q RMS ~/.bashrc)
then
	cat <<EOF >>~/.bashrc
	# Activate RMS
	cd ~/source/RMS
	source ~/vRMS/bin/activate
EOF
fi
if ( ! command -v pcmanfm &> /dev/null ) 
then
	echo -e "Install pcmanfm file manager ?\n"
	while true; do
		read -p  "Y/N: " Ans
		case $Ans in 
		[yY]* ) sudo apt install -y pcmanfm; break;;
		[nN] ) break ;;
		esac
		done
fi
if (! command -v mousepad &> /dev/null )
then
	echo -e "Install mousepad ?\n"
	while true; do
   		read -p  "Y/N: " Ans
       	case $Ans in 
           	[yY]* ) sudo apt install -y mousepad; break;;
           	[nN] ) break ;;
           	esac
		done

fi

if ( ! command -v anydesk &> /dev/null ) 
then
	echo -e "Install AnyDesk ?"
	echo -e "This will require sudo privileges \n"
	while true; do
   		read -p  "Y/N: " Ans
       	case $Ans in 
       	[yY]* ) 
		wget -qO - https://keys.anydesk.com/repos/DEB-GPG-KEY | sudo apt-key add -
		echo "deb http://deb.anydesk.com/ all main" > anydesk-stable.list
		sudo mv anydesk-stable.list /etc/apt/sources.list.d/
		sudo apt update
		sudo apt install -y anydesk
		break;;
       	[nN] ) break ;;
       	esac
	done
fi

echo -e "\n\nStation configuration complete\n\n"
user=`id -u -n`

# Optionaly create a crontab entry for the user to automate regular RMS updates I've chosen midday as a default but this can easily be changed
echo -e "Do you wish to schedule a regular cron job to stop all running instances,"
echo -e "perform an update of RMS, and finally restart all configured instances? "
while  true; do
   	read -p  "Y/N: " Ans
    case $Ans in 
    [yY]* ) 
	echo " The following template entry will schedule the action to run @3pm every Sunday"
	echo " feel free to edit the line if you want to alter the timing and/or frequency, when you are happy with"
	echo "the schedule, just hit return"
	echo " If you want some examples of the required format check out  https://crontab.guru/examples.html"
	echo "0 15 * * SUN /home/${user}/source/RMS/Scripts/MultiCamLinux/GRMSUpdate.sh" | read Schedule
	if sudo test -f "/var/spool/cron/crontabs/$user"   # does user have an existing crontab?
       	then
   		if sudo grep -q RMSUpdater /var/spool/cron/crontabs/$user # check there isn't an existing entry
       	then
    	   	echo -e "\nNot modifying ${user}'s crontab  -that user has an existing entry\n"
		else
   			read -e -i "0 15 * * SUN" Schedule
			# can't edit users crontab in-place without messing with dir permissions...
   			sudo mv /var/spool/cron/crontabs/$user $user.tmp
   			set -o noglob
   			sudo echo "$Schedule /home/$user/source/RMS/Scripts/MultiCamLinux/GRMSUpdater.sh" >> ${user}.tmp 
   			set +o noglob
   			sudo mv ${user}.tmp /var/spool/cron/crontabs/${user}
   			echo -e "\n User  ${user}'s crontab has been updated, if you subsequently wish to edit it you can do so"
   			echo -e "by issuing the following cmd in a terminal-  crontab -e\n"
	fi
else
	read -e -i "0 15 * * SUN" Schedule
	set -o noglob
	# user does not have a crontab so create one with the standard boilerplate stuff
	touch ${user}.tmp
	cat <<- EOF >${user}.tmp
	# Edit this file to introduce tasks to be run by cron.
	# 
	# Each task to run has to be defined through a single line
	# indicating with different fields when the task will be run
	# and what command to run for the task
	# 
	# To define the time you can provide concrete values for
	# minute (m), hour (h), day of month (dom), month (mon),
	# and day of week (dow) or use '*' in these fields (for 'any').
	# 
	# Notice that tasks will be started based on the cron's system
	# daemon's notion of time and timezones.
	# 
	# Output of the crontab jobs (including errors) is sent through
	# email to the user the crontab file belongs to (unless redirected).
	# 
	# For example, you can run a backup of all your user accounts
	# at 5 a.m every week with:
	# 0 5 * * 1 tar -zcf /var/backups/home.tgz /home/
	# 
	# For more information see the manual pages of crontab(5) and cron(8)
	# 
	# m h  dom mon dow   command
	$Schedule /home/$user/source/RMS/Scripts/MultiCamLinux/GRMSUpdater.sh
	EOF

	set -o noglob
	sudo mv ${user}.tmp /var/spool/cron/crontabs/${user}
	echo -e "\nA new crontab has been installed for ${user}, if you subsequently wish to change it in any way"
	echo -e "you can edit it in a terminal with the cmd - crontab -e\n"
	fi
	break ;;
	[nN]* ) break ;;
	esac		
	done

# Set up shortcut for CMNbinViewer
cat <<- EOF > ~/Desktop/CMNbinViewer.desktop
[Desktop Entry]
Name=CMNbinViewer
Type=Application
Exec=/home/${user}/source/RMS/Scripts/CMNbinViewer_env.sh
Hidden=false
NoDisplay=false
Icon=/home/${user}/source/cmn_binviewer/icon.png
EOF
# set the 'allow launching' feature
chmod -x ~/Desktop/CMNbinViewer.desktop
gio set ~/Desktop/CMNbinViewer.desktop metadata::trusted true
chmod +x ~/Desktop/CMNbinViewer.desktop
if grep -Fqv "UTC" /etc/timezone
then
sudo timedatectl set-timezone UTC
fi
# Fix AnyDesk issue with Wayland ... requires sudo
if grep -q '#Way' /etc/gdm3/custom.conf   
then
	sudo sed -i 's/#\(WaylandEnable.*\)/\1/g' /etc/gdm3/custom.conf
fi
exit
