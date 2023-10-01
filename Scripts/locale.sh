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
echo "This script will allow you to change the default locale for the downloaded "
echo "RMS RPi pre-built image."
echo "It will create new default $HOME directories named as per your locale and it will sym-link"
echo "the Desktop equivelent entries to your locale version of Desktop whilst leaving the originals as-is"
echo "In this way should you require support from an English speaking person then the not only are the original"
echo "Desktop files still present  but is is possible to revert locale back to en.CA so that system error messages are rendered in English"
echo "The process of changing locales is a two step process -"
echo "Firstly via a chooser you select the languange variant you want, it is recommended to choose UTF-8 variants where possible"
echo "Secondly you select  of those compiled locales, which shoul dbe set as default"
echo "So if you need to revert back to en.CA you can merely select <OK> for the 1st stage and then select en.CA at the 2nd prompt since the original"
echo "en.CA locale i=will already be present and compiled in the default RPi image"
echo "The command to change your locale manually is -"
echo -e "\nsudo dpkg-reconfigure locales\n"
read -p "hit <CR> to continue..."
sudo dpkg-reconfigure locales
#
# let user choose their locale, ensure they also select their chosen default....
#
# grab the users choice via the LANG variable in /etc/default/locale
UserLang=$(awk 'BEGIN { FS="=" } /LANG=/ {print $2}' /etc/default/locale)
# set the new locale as default
sudo update-locale LANG=${UserLang}  LANGUAGE=${UserLang} LC_ALL=${UserLang}
# update locale for this session
export LANG=${UserLang}
export LANGUAGE=${UserLang}
export LC_ALL=${UserLang}
cd ~/
# create new locale named $HOME dirs
xdg-user-dirs-update --force
# SymLink existing Desktop files to new locale dir
for File in Desktop/*;
      do 
      ln -s ~/"$File" "$(xdg-user-dir DESKTOP)"/"$(basename "$File")"
      done

