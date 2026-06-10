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
# Universal station tool for any Linux platform, Raspberry Pi included.
# Merged from add_Pi_Station.sh and add_GStation.sh, which are kept as
# compatibility wrappers around this script.
#
# What it does:
#  - converts a legacy single-camera setup (config files in ~/source/RMS,
#    data directly in ~/RMS_data) into the per-station layout under
#    ~/source/Stations/<ID> and ~/RMS_data/<ID>
#  - creates new stations, either interactively or in bulk from a CSV file
#
# Usage: add_Station.sh [RMS_data_path] [stations.csv]
#
# The CSV header row names the columns: station_id (required), camera_ip
# (spliced into the rtsp device URL), and any other .config key, which is
# applied verbatim to that station's config. With no CSV argument the
# script uses ~/source/RMS/stations.csv if present, otherwise it prompts
# for station IDs interactively. A sample to copy and edit is provided in
# ~/source/RMS/stations_template.csv.
#
# On Raspberry Pi platforms a recommended camera limit applies (1 camera,
# or 4 on a Pi5 or later). The limit is advisory - it reflects what has
# been tested - and can be overridden at the prompt.

# ---------------------------------------------------------------- helpers

# Strip leading/trailing whitespace (including CR from Windows-edited CSVs)
trim() {
    local s="$*"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

uppercase() {
    printf '%s' "$*" | tr '[:lower:]' '[:upper:]'
}

count_stations() {
    find ~/source/Stations -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
}

# Build the Exec= line for a .desktop entry using the detected terminal.
# The gnome-terminal StartCapture profile is optional - create one to
# customise the capture windows; without it the default profile is used.
term_exec() {   # $1 = window title, $2 = command
    if [[ "$TERMINAL" == "gnome-terminal" ]]; then
        printf 'gnome-terminal --profile=StartCapture --title=%s -- bash -c "%s"' "$1" "$2"
    else
        printf 'lxterminal --title=%s -e "%s"' "$1" "$2"
    fi
}

# Create a station directory with config, mask and launch entries.
# Returns 1 if the station already exists.
create_station() {
    local item="$1"
    local config_file=~/source/Stations/${item}/.config
    local entry

    if [[ -d ~/source/Stations/${item} ]]; then
        echo -e "\nNot creating station ${item} - it already exists"
        return 1
    fi

    echo "Creating station ${item}..."
    mkdir -p ~/source/Stations/${item}
    mkdir -p "${RMS_data}/${item}"
    mkdir -p ~/.config/autostart "${Desktop}"

    # Copy config template
    cp ~/source/RMS/.config ~/source/Stations/${item}

    # Copy mask file
    cp ~/source/RMS/mask.bmp ~/source/Stations/${item} 2>/dev/null

    # Launch entry for the station. On the Pi images, FirstRun runs at boot
    # and starts captures through RMS_StartCapture_MCP.sh, so only a Desktop
    # launcher is created there; elsewhere the entry goes into XDG autostart
    # with a Desktop symlink, so captures start at login. (A PC may carry the
    # autorun flag from running FirstRun, but nothing launches FirstRun at
    # boot on a PC, hence the additional Pi check.)
    if [[ -f ~/.rmsautorunflag && $RECOMMENDED -gt 0 ]]; then
        entry="${Desktop}/${item}_StartCapture.desktop"
    else
        entry=~/.config/autostart/${item}_StartCapture.desktop
    fi

    cat <<- EOF > "$entry"
	[Desktop Entry]
	Name=${item}-StartCapture
	Type=Application
	Exec=$(term_exec "${item}" "~/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${item}")
	Hidden=false
	NoDisplay=false
	Icon=${TERMINAL}
	EOF
    chmod +x "$entry"

    if [[ "$entry" != "${Desktop}/"* ]]; then
        ln -sf "$entry" "${Desktop}/${item}_StartCapture.desktop"
        gio set "${Desktop}/${item}_StartCapture.desktop" metadata::trusted true 2>/dev/null
        chmod +x "${Desktop}/${item}_StartCapture.desktop"
    fi

    # ShowLiveStream desktop shortcut
    cat <<- EOF > "${Desktop}/${item}-ShowLiveStream.desktop"
	[Desktop Entry]
	Name=${item}-ShowLiveStream
	Type=Application
	Exec=$(term_exec "Stream-${item}" "~/source/RMS/Scripts/MultiCamLinux/LiveStream.sh ${item}")
	Hidden=false
	NoDisplay=false
	Icon=${TERMINAL}
	EOF
    gio set "${Desktop}/${item}-ShowLiveStream.desktop" metadata::trusted true 2>/dev/null
    chmod +x "${Desktop}/${item}-ShowLiveStream.desktop"

    # Customise the config
    sed -i "s/^stationID:.*$/stationID: $item/g" "${config_file}"
    sed -i "s,^data_dir:.*$,data_dir: ${RMS_data}/${item},g" "${config_file}"

    return 0
}

# Convert a legacy single-camera setup to the per-station layout by copying
migrate_legacy() {

cat <<EOF

Multiple cameras are supported by giving each camera its own copies of the
configuration files under ~/source/Stations/<station_ID>.

Your configured station ${DefStation} will get copies of:
- .config
- platepar_cmn2010.cal
- mask.bmp
in ~/source/Stations/${DefStation}

Nothing in ~/source/RMS is modified, and camera_settings.json is shared
by all stations from there.

Any previously captured data will be moved from
${RMS_data}
to
${RMS_data}/${DefStation}
EOF

    read -n1 -r -p 'Press ENTER to continue with the conversion, or any other key to skip it: ' key
    echo ""
    if [[ "$key" != "" ]]; then
        echo "Skipping the conversion"
        return
    fi

    create_station "${DefStation}" || return

    # the .config and mask copies are made by create_station
    if [[ -e ~/source/RMS/platepar_cmn2010.cal ]]; then
        cp ~/source/RMS/platepar_cmn2010.cal ~/source/Stations/${DefStation}/
    fi

    # move any existing captured data into the station folder
    find "${RMS_data}" -mindepth 1 -maxdepth 1 ! -name "${DefStation}" -exec mv -t "${RMS_data}/${DefStation}/" {} +

    # Pi image cosmetics - skipped when the files are not present
    if [[ -f ~/.conkyrc1 ]]; then
        # tweak conky station title and data source path for log data
        sed -i 's/\(source\)\/RMS/\1\/Stations\/'"$DefStation"'/g' ~/.conkyrc1
        sed -i "s/\(.*RMS_data\)/\1\/${DefStation}/g" ~/.conkyrc1
    fi

    # cleanup single-camera Desktop shortcuts from the Pi image
    if [[ -f ~/Desktop/RMS_FirstRun.sh ]]; then
        rm -f ~/Desktop/CMNbinViewer.sh ~/Desktop/RMS_ShowLiveStream.sh
        rm -f ~/Desktop/RMS_StartCapture.sh ~/Desktop/RMS_config.txt
        rm -f ~/Desktop/TunnelIPCamera.sh ~/Desktop/DownloadOpenVPNconfig.sh
    fi

    echo ""
    echo "Conversion of station ${DefStation} complete"
}

# ------------------------------------------------------------------ main

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
    echo "A sample to copy and edit is provided in ~/source/RMS/stations_template.csv"
    exit 1
fi

# Detect the platform. On Raspberry Pi a recommended camera limit applies:
# 1 camera, or 4 on a Pi5 or later (the tested ceiling). The limit is
# advisory and can be overridden at the prompt.
PI_MODEL_FILE="${RMS_PI_MODEL_FILE:-/sys/firmware/devicetree/base/model}"
RECOMMENDED=0   # 0 = no recommendation (generic Linux)
if [[ -f "$PI_MODEL_FILE" ]]; then
    model=$(tr -d '\0' < "$PI_MODEL_FILE")
    if [[ "$model" == *'Raspberry'* ]]; then
        pi_num=$(printf '%s' "$model" | grep -oE '[0-9]+' | head -1)
        if [[ -n "$pi_num" && "$pi_num" -ge 5 ]]; then
            RECOMMENDED=4
        else
            RECOMMENDED=1
        fi
        echo "Detected: ${model} (recommended maximum: ${RECOMMENDED} camera(s))"
    fi
fi

# Detect the terminal emulator for launch entries
if [[ "${XDG_CURRENT_DESKTOP:-}" == *GNOME* ]] && command -v gnome-terminal >/dev/null 2>&1; then
    TERMINAL=gnome-terminal
elif command -v lxterminal >/dev/null 2>&1; then
    TERMINAL=lxterminal
elif command -v gnome-terminal >/dev/null 2>&1; then
    TERMINAL=gnome-terminal
else
    TERMINAL=lxterminal
    echo "Warning: neither lxterminal nor gnome-terminal was found, launch entries will use lxterminal"
fi

# Get user's desktop directory
Desktop=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")

DefStation=$(uppercase "$(awk '/^stationID:/ {print $2}' ~/source/RMS/.config 2>/dev/null)")

# A configured legacy single-camera setup is offered conversion first
if [[ ! -d ~/source/Stations && -n "$DefStation" && "$DefStation" != "XX0001" ]]; then
    migrate_legacy
elif [[ -f ~/.rmsautorunflag && "$DefStation" == "XX0001" && ! -d ~/source/Stations ]]; then
    echo "Please run RMS_FirstRun and configure your 1st station,"
    echo "then run add_Station again to convert it or add more cameras."
    exit
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

    No_Stations=$(tail -n +2 "${CSV_FILE}" | grep -c -v '^[[:space:]]*$')
    echo -e "\nStations to add from CSV: ${No_Stations}"

    # Warn when the CSV would exceed the recommended camera limit
    if [[ $RECOMMENDED -gt 0 && $(( $(count_stations) + No_Stations )) -gt $RECOMMENDED ]]; then
        echo ""
        echo "This would exceed the recommended maximum of ${RECOMMENDED} camera(s) for this platform"
        echo "(the tested limit - more cameras may drop frames during capture and processing)."
        read -n1 -r -p 'Press Y to continue anyway, any other key to quit: ' key
        echo ""
        if [[ "$key" != "y" && "$key" != "Y" ]]; then
            exit
        fi
    fi

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
        # Advise when the recommended camera limit is reached, allow override
        if [[ $RECOMMENDED -gt 0 && $(( $(count_stations) + ${#Station[@]} )) -ge $RECOMMENDED ]]; then
            echo ""
            echo "The recommended maximum of ${RECOMMENDED} camera(s) for this platform is configured"
            echo "(the tested limit - more cameras may drop frames during capture and processing)."
            read -n1 -r -p 'Press ENTER to finish, or A to add another camera anyway: ' key
            echo ""
            if [[ "$key" != "a" && "$key" != "A" ]]; then
                break
            fi
        fi
        read -p "Enter station ID, <cr> to end: " this_Station
        this_Station=$(uppercase "$(trim "$this_Station")")
        if [[ -z "$this_Station" ]]; then
            break
        fi
        Station+=("$this_Station")
    done

    if [[ ${#Station[@]} -gt 0 ]]; then
        echo -e "\nNew stations to add -"
        printf '%s\n' "${Station[@]}"
    fi

    for item in "${Station[@]}"; do
        create_station "$item" && echo "Added station ${item}"
    done

fi

# -------------------------------------------------- per-platform settings

total=$(count_stations)

if [[ $total -gt 1 ]]; then
    for dir in ~/source/Stations/*/; do
        # scale the reserved free space with the number of stations
        sed -i "s/^extra_space_gb:.*$/extra_space_gb: $(( total * 20 ))/g" "${dir}/.config"
        # disable the daily post processing reboot, it would kill the other captures
        sed -i "s/^\(reboot_after_processing:\).*/\1 false/g" "${dir}/.config"
    done

    # remove comment from last line of wayfire.ini to enable window cascade (Pi image)
    if [[ -f ~/.config/wayfire.ini ]]; then
        sed -i s/#mode/mode/ ~/.config/wayfire.ini
    fi
fi

# Point the Desktop StartCapture entry at the multi-camera launcher, which
# starts every configured station. The legacy entry it replaces reads
# ~/source/RMS/.config, which after a conversion is just the template.
# On the Pi images FirstRun also runs this launcher at boot.
rm -f "${Desktop}/RMS_StartCapture.sh"
ln -sf ~/source/RMS/Scripts/MultiCamLinux/Pi/RMS_StartCapture_MCP.sh "${Desktop}/RMS_StartCapture.sh"

# Generate SSH keys if not present
if [[ ! -f ~/.ssh/id_rsa ]]; then
    ssh-keygen -t rsa -f ~/.ssh/id_rsa -q -P ""
    cp ~/.ssh/id_rsa.pub "${Desktop}/"
    echo "SSH keys generated in ~/.ssh"
    echo "Your new id_rsa.pub public key file is now placed on the Desktop."
    echo "Be sure to send a copy of this file to Denis"
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

# Create Desktop shortcut for editing the station .config files
cat << 'EOF' > "${Desktop}/Edit_configs"
#!/bin/bash

echo ""
echo "One by one, each config file under ~/source/Stations will open for editing..."
read -n1 -r -p 'Press ENTER to proceed, any other key to quit: ' key
if [[ "${key}" != "" ]]; then
    exit
fi

editor=""
for e in mousepad leafpad gedit gnome-text-editor nano; do
    if command -v "$e" >/dev/null 2>&1; then
        editor="$e"
        break
    fi
done

for Dir in ~/source/Stations/*/
do
    Station=$(basename "${Dir}")
    echo ""
    echo "Editing .config file for ${Station}"
    read -n1 -r -p 'Press ENTER to edit, any other key to skip: ' key
    if [[ "${key}" = "" ]]; then
	"$editor" ~/source/Stations/${Station}/.config
    fi
done

echo ""
echo "Done editing .config files"
sleep 2
EOF
chmod +x "${Desktop}/Edit_configs"

echo -e "\nStation configuration complete"
