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
# Version 1.7   - stations can be created in bulk from a stations.csv file,
#                 desktop entries use gnome-terminal instead of lxterminal,
#                 interactive prompts remain the fallback when no CSV is given
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

# Prevent running this script on a Raspberry Pi
file=/sys/firmware/devicetree/base/model
if [[ -f "$file" ]]; then
    contents=$(tr -d '\0' < $file)
    if [[ $contents == 'Raspberry'* ]]; then
        echo "The add_GStation.sh script should not be used on Raspberry Pi."
        echo "Please use add_Pi_Station.sh to add cameras on a Pi5."
        exit 1
    fi
fi

# Get user's desktop directory
Desktop=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")

# Strip leading/trailing whitespace (including CR from Windows-edited CSVs)
trim() {
    local s="$*"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Create a station directory with config, mask and desktop entries.
# Returns 1 if the station already exists.
create_station() {
    local item="$1"
    local config_file=~/source/Stations/${item}/.config

    if [[ -d ~/source/Stations/${item} ]]; then
        echo -e "\nNot creating station ${item} - it already exists"
        return 1
    fi

    echo "Creating station ${item}..."
    mkdir -p ~/source/Stations/${item}
    mkdir -p "${RMS_data}/${item}"

    # Copy config template
    cp ~/source/RMS/.config ~/source/Stations/${item}

    # Create autostart .desktop entry using gnome-terminal.
    # The StartCapture profile is optional - create one to customise the
    # capture window; without it gnome-terminal uses the default profile.
    cat <<- EOF > ~/.config/autostart/${item}_StartCap.desktop
	[Desktop Entry]
	Name=${item}_StartCap
	Type=Application
	Exec=gnome-terminal --profile=StartCapture --title=${item} -- bash -c "~/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${item}"
	Hidden=false
	NoDisplay=false
	Icon=gnome-terminal
	EOF

    chmod +x ~/.config/autostart/${item}_StartCap.desktop

    # Create Desktop symlink for StartCapture
    ln -sf ~/.config/autostart/${item}_StartCap.desktop "${Desktop}/${item}_StartCap.desktop"
    gio set "${Desktop}/${item}_StartCap.desktop" metadata::trusted true 2>/dev/null
    chmod +x "${Desktop}/${item}_StartCap.desktop"

    # Copy mask file
    cp ~/source/RMS/mask.bmp ~/source/Stations/${item} 2>/dev/null

    # Create ShowLiveStream desktop shortcut using gnome-terminal
    cat <<- EOF > "${Desktop}/Show_LiveStream-${item}.desktop"
	[Desktop Entry]
	Name=${item}-ShowLiveStream
	Type=Application
	Exec=gnome-terminal --profile=StartCapture --title=Stream-${item} -- bash -c "~/source/RMS/Scripts/MultiCamLinux/LiveStream.sh ${item}"
	Hidden=false
	NoDisplay=false
	Icon=gnome-terminal
	EOF

    gio set "${Desktop}/Show_LiveStream-${item}.desktop" metadata::trusted true 2>/dev/null
    chmod +x "${Desktop}/Show_LiveStream-${item}.desktop"

    # Apply default config customizations
    sed -i "s/^stationID:.*$/stationID: $item/g" "${config_file}"
    sed -i "s,^data_dir:.*$,data_dir: ${RMS_data}/${item},g" "${config_file}"
    sed -i "s/^extra_space_gb:.*$/extra_space_gb: $(( ${No_Stations} * 20 ))/g" "${config_file}"
    sed -i "s/^\(reboot_after_processing:\).*/\1 false/g" "${config_file}"

    return 0
}

# Parse arguments: [RMS_data_path] [stations.csv]
# Smart detection: if single arg ends in .csv, treat as CSV file
RMS_data=~/RMS_data
CSV_FILE=""

if [[ "$#" -eq 0 ]]; then
    # No args: use the default CSV if one exists
    if [[ -f ~/source/RMS/stations.csv ]]; then
        CSV_FILE=~/source/RMS/stations.csv
    fi
elif [[ "$#" -eq 1 ]]; then
    # Single arg: check if it's a CSV file or a directory
    if [[ "${1}" == *.csv ]]; then
        CSV_FILE="${1}"
    else
        RMS_data="${1}"
        if [[ -f ~/source/RMS/stations.csv ]]; then
            CSV_FILE=~/source/RMS/stations.csv
        fi
    fi
else
    # Two args: first is RMS_data, second is CSV
    RMS_data="${1}"
    CSV_FILE="${2}"
fi

# A CSV that was explicitly requested must exist
if [[ -n "${CSV_FILE}" && ! -f "${CSV_FILE}" ]]; then
    echo "Error: CSV file not found: ${CSV_FILE}"
    echo "Usage: $0 [RMS_data_path] [stations.csv]"
    exit 1
fi

# Ensure required directories exist
mkdir -p ~/source/Stations
mkdir -p "${RMS_data}"
mkdir -p ~/.config/autostart

if [[ -n "${CSV_FILE}" ]]; then

    echo "Reading stations from: ${CSV_FILE}"

    # Read CSV header to get column names
    IFS=',' read -r -a HEADERS < "${CSV_FILE}"

    # Trim whitespace from headers
    for i in "${!HEADERS[@]}"; do
        HEADERS[$i]=$(trim "${HEADERS[$i]}")
    done

    # Find required column indices
    station_id_col=-1
    camera_ip_col=-1
    for i in "${!HEADERS[@]}"; do
        case "${HEADERS[$i]}" in
            station_id|stationID) station_id_col=$i ;;
            camera_ip) camera_ip_col=$i ;;
        esac
    done

    if [[ $station_id_col -eq -1 ]]; then
        echo "Error: CSV must have a 'station_id' or 'stationID' column"
        exit 1
    fi

    # Count stations for extra_space_gb calculation
    No_Stations=$(tail -n +2 "${CSV_FILE}" | grep -c -v '^[[:space:]]*$')

    echo -e "\nStations to add from CSV: ${No_Stations}"

    # Process each data row (the || keeps a final line without a trailing newline)
    tail -n +2 "${CSV_FILE}" | while IFS=',' read -r -a VALUES || [[ ${#VALUES[@]} -gt 0 ]]; do
        # Skip empty lines
        [[ ${#VALUES[@]} -eq 0 ]] && continue

        # Trim whitespace from values
        for i in "${!VALUES[@]}"; do
            VALUES[$i]=$(trim "${VALUES[$i]}")
        done

        item="${VALUES[$station_id_col]}"
        [[ -z "$item" ]] && continue

        create_station "$item" || continue

        config_file=~/source/Stations/${item}/.config

        # Apply camera_ip to device URL if column exists
        if [[ $camera_ip_col -ne -1 && -n "${VALUES[$camera_ip_col]}" ]]; then
            camera_ip="${VALUES[$camera_ip_col]}"
            # Replace IP address in the device URL (matches IP pattern in rtsp:// URL)
            sed -i -E "s|^(device:.*rtsp://)[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|\1${camera_ip}|g" "${config_file}"
            echo "  Set camera IP: ${camera_ip}"
        fi

        # Apply any additional CSV columns that match .config keys
        for i in "${!HEADERS[@]}"; do
            header="${HEADERS[$i]}"
            value="${VALUES[$i]}"

            # Skip station_id, camera_ip (already handled), and empty values
            [[ "$header" == "station_id" || "$header" == "stationID" || "$header" == "camera_ip" ]] && continue
            [[ -z "$value" ]] && continue

            # Check if this header exists as a key in .config
            if grep -q "^${header}:" "${config_file}"; then
                sed -i "s|^${header}:.*$|${header}: ${value}|g" "${config_file}"
                echo "  Set ${header}: ${value}"
            fi
        done

        echo "Added station ${item}"
    done

else

    # No CSV available - prompt for station IDs interactively
    declare -a Station
    while :
    do
        read -p "Enter station ID, <cr> to end: " this_Station
        this_Station=$(trim "${this_Station^^}")
        if [[ -z $this_Station ]]; then
            break
        fi
        Station+=("$this_Station")
    done

    No_Stations=${#Station[@]}
    echo -e "\nNew stations to add -"
    printf '%s\n' "${Station[@]}"

    for item in "${Station[@]}"; do
        create_station "$item" && echo "Added station ${item}"
    done

fi

# Generate SSH keys if not present
if [[ ! -f ~/.ssh/id_rsa ]]; then
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -P ""
    echo "SSH keys generated in ~/.ssh"
fi

# Add RMS activation to .bashrc if not present
if ! grep -q "source ~/vRMS/bin/activate" ~/.bashrc; then
    cat <<EOF >> ~/.bashrc

# Activate RMS
cd ~/source/RMS
source ~/vRMS/bin/activate
EOF
fi

# Create CMNbinViewer desktop shortcut
cat <<- EOF > "${Desktop}/CMNbinViewer.desktop"
[Desktop Entry]
Name=CMNbinViewer
Type=Application
Exec=${HOME}/source/RMS/Scripts/CMNbinViewer_env.sh
Hidden=false
NoDisplay=false
Icon=${HOME}/source/RMS/Scripts/MultiCamLinux/icon.png
EOF

gio set "${Desktop}/CMNbinViewer.desktop" metadata::trusted true 2>/dev/null
chmod +x "${Desktop}/CMNbinViewer.desktop"

echo -e "\nStation configuration complete"
