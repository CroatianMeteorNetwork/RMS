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
#
# Version 1.1
# Changes - added station argument
#

if [[  -z "$1" ]]	# called with no args
then
	echo " No Station directory specified, quitting now"
	sleep 3
	exit
fi
cd ~/source/Stations/$1
echo "Starting RMS live stream..."
source ~/vRMS/bin/activate
python -m Utils.ShowLiveStream

