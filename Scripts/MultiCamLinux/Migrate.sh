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
# Version 1.0
# Initial release
#
#
# This script automates the trasnsfer of the 3  RMS station configuration files and places these into the correct Stations folder on this host
# It prompts the user for the target hostname and copies this hosts public to the targert host to enable passwordless authentication host using ssh keys, 
# and if that fails will prompt for a password to install this hosts public key on the target host.
# Since presumably this is a new host it will likely not have it's public keys loaded on your target host and also the host id keys will also not be known to 
# this host so you will be prompted to accept and trust each new host..
# It then will copy those config files to a temporary dir, parse the .config.tmp for the hostname and check if that station exists within ~/source/Stations
# and if it does then it will transfer them all to the appropriate Stations dir, it then finaly edits in-place the .config to update the new RMS_data dir.
# The script will continue to prompt for additional hosts and repeat the process until the user enters a <CR> at the host prompt


# create a migrate dir to store temp files in so as to ensure we don't leave any bare files lying around should the user hit CTRL-C...
if [[ !  -d migrate ]]
        then
	mkdir migrate
fi
cd migrate

# check this host has a public key  -if the user has previously run the add_GStation script it ought to exist but.....
if [[ ! -f ~/.ssh/id_rsa ]]
        then
        ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -P ""
        echo "SSH keys successfully generated for this host in ~/.ssh"
fi

read -p  "Enter the RPi's IP address: " TargetIP
	if [[ -z $TargetIP ]]
	then
		break
	fi
# we assume the default user for the host is pi
RmsUser=pi
if ( ! ping -c1 -W1 ${TargetIP} >/dev/null 2>&1 ) # is host reachable
	then
	if ( nc -z -w 3 ${TargetIP} 22 2>/dev/null ) # host is reachable and port 22 SSH open
        	then
                continue
                else 
                echo -e "\nhost ${TargetIP} is reachable , however port 22 is not open. - have you enabled the SSH daemon on this host?"
                exit
	fi
	echo -e "\nHost is not reachable"
fi

	echo -e "\n\nIf this hosts keys are not present on $TargetIP, you will be asked to add the hosts unique fingerprint to your .ssh/known_hosts file - when prompted answer 'yes' "
	echo -e "You will then be prompted for user pi's password to copy the key to the authorized_keys file located in .ssh on $TargetHost\n\n"
	ssh-copy-id $RmsUser@$TargetIP
        scp $RmsUser@$TargetIP:~/source/RMS/.config .
        scp $RmsUser@$TargetIP:~/source/RMS/mask.bmp .
        scp $RmsUser@$TargetIP:~/source/RMS/platepar_cmn2010.cal .
        scp $RmsUser@$TargetIP:~/.ssh/id_rsa .

# check the station  has been configured ...
	Station=$( awk '/stationID:/ {print $2}' .config )
	if [[ ! -f ~/source/Stations/$Station ]]
	echo -e "\n\n Found station ${Station} ! \n"
	then
		mv .config ~/source/Stations/$Station/ 
		mv mask.bmp  ~/source/Stations/$Station/ 
		mv platepar_cmn2010.cal ~/source/Stations/$Station/
		mv id_rsa ~/.ssh/${Station}_id_rsa # take the RPi's private key and place it in ~/.ssh with the station name prepending it
		sed -i "s/data_dir.*$/data_dir: ~\/RMS_data\/${Station}/g" ~/source/Stations/${Station}/.config
		sed -i "s/\(.*key:\).*/\1 ~\/.ssh\/${Station}_id_rsa/g" ~/source/Stations/${Station}/.config # update path to this stations unique key
		echo -e "\n\nStations ${Station}'s .config had been moved and RMS_data location updated, mask.bmp and platepar_cmn2010.cal have been moved unchanged"
		echo -e "and its private key have been placed in ~/.ssh and renamed ${Station}_id_rsa\n"
	else
	echo -e "\n$Station has not been configured on this host\n"
	# cleanup the migrate dir in case a mv failed...
	rm  .config mask.bmp platepar_cmn2010.cal
	fi	
cd -
rmdir migrate
