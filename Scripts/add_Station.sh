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
#    data directly in the data directory) into the per-station layout under
#    ~/source/Stations/<ID> and <data directory>/<ID>
#  - creates new stations, either interactively or in bulk from a CSV file
#
# Usage: add_Station.sh [RMS_data_path] [stations.csv]
#
# RMS_data_path defaults to the data_dir of ~/source/RMS/.config, or
# ~/RMS_data if that is not set.
#
# The CSV header row names the columns: station_id (required), camera_ip
# (spliced into the rtsp device URL), and any other .config key, which is
# applied verbatim to that station's config. With no CSV argument the
# script uses ~/source/RMS/stations.csv if present, otherwise it prompts
# for station IDs interactively. A sample to copy and edit is provided in
# ~/source/RMS/Scripts/stations_template.csv.
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

# Print the value of a .config key, without any inline ; comment
get_config() {   # $1 = key, $2 = config file
    KEY="$1" awk '
        index($0, ENVIRON["KEY"] ":") == 1 {
            v = substr($0, length(ENVIRON["KEY"]) + 2)
            sub(/[ \t];.*$/, "", v)
            gsub(/^[ \t]+|[ \t\r]+$/, "", v)
            print v
            exit
        }' "$2" 2>/dev/null
}

# Set the value of a .config key. The value is taken literally - unlike a
# sed replacement, where & | and \ would corrupt it (e.g. in a device URL).
# Returns 1 if the key is not in the file.
set_config() {   # $1 = key, $2 = value, $3 = config file
    local tmp
    tmp=$(mktemp) || return 1
    if KEY="$1" VALUE="$2" awk '
        index($0, ENVIRON["KEY"] ":") == 1 { print ENVIRON["KEY"] ": " ENVIRON["VALUE"]; found = 1; next }
        { print }
        END { exit !found }' "$3" > "$tmp"; then
        cat "$tmp" > "$3"
        rm -f "$tmp"
        return 0
    fi
    rm -f "$tmp"
    return 1
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
    # and starts captures through RMS_StartCapture.sh, so only a Desktop
    # launcher is created there; elsewhere the entry goes into XDG autostart
    # with a Desktop symlink, so captures start at login. (A PC may carry the
    # autorun flag from running FirstRun, but nothing launches FirstRun at
    # boot on a PC, hence the additional Pi check.)
    if [[ -f ~/.rmsautorunflag && $IS_PI -eq 1 ]]; then
        entry="${Desktop}/${item}_StartCapture.desktop"
    else
        entry=~/.config/autostart/${item}_StartCapture.desktop
    fi

    # Exec= paths use ${HOME}: the desktop entry spec does not expand ~
    cat <<- EOF > "$entry"
	[Desktop Entry]
	Name=${item}-StartCapture
	Type=Application
	Exec=$(term_exec "${item}" "${HOME}/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${item}")
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
	Exec=$(term_exec "Stream-${item}" "${HOME}/source/RMS/Scripts/MultiCamLinux/LiveStream.sh ${item}")
	Hidden=false
	NoDisplay=false
	Icon=${TERMINAL}
	EOF
    gio set "${Desktop}/${item}-ShowLiveStream.desktop" metadata::trusted true 2>/dev/null
    chmod +x "${Desktop}/${item}-ShowLiveStream.desktop"

    # Customise the config
    set_config stationID "${item}" "${config_file}"
    set_config data_dir "${RMS_data}/${item}" "${config_file}"

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
        # Adding cameras without converting would stop the capture of the
        # legacy camera, which RMS_StartCapture.sh only runs while no
        # station exists
        echo "Skipping the conversion - nothing was changed."
        echo "To convert later, or to add more cameras, run this command from a terminal:"
        echo "  ~/source/RMS/Scripts/add_Station.sh"
        return 1
    fi

    create_station "${DefStation}" || return 1

    # the .config and mask copies are made by create_station
    if [[ -e ~/source/RMS/platepar_cmn2010.cal ]]; then
        cp ~/source/RMS/platepar_cmn2010.cal ~/source/Stations/${DefStation}/
    fi

    # move any existing captured data into the station folder (lost+found
    # stays where it is when the data directory is a mount point)
    find "${RMS_data}" -mindepth 1 -maxdepth 1 ! -name "${DefStation}" ! -name lost+found -exec mv -t "${RMS_data}/${DefStation}/" {} +

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
    converted=1
}

# ------------------------------------------------------------------ main

# Parse arguments: [RMS_data_path] [stations.csv]
# Smart detection: if single arg ends in .csv, treat as CSV file

# The data directory defaults to the data_dir of the root config (expanded
# as ConfigReader does), so data kept elsewhere, e.g. on an external drive,
# is converted where it is instead of being left behind
RMS_data=$(get_config data_dir ~/source/RMS/.config)
if [[ -z "$RMS_data" ]]; then
    RMS_data=~/RMS_data
elif [[ "$RMS_data" == "~"* ]]; then
    RMS_data="${RMS_data#\~}"
    RMS_data="${HOME}/${RMS_data#/}"
elif [[ "$RMS_data" != /* ]]; then
    # relative paths are resolved from ~/source/RMS, where capture runs
    RMS_data=~/source/RMS/${RMS_data}
fi
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
    echo "A sample to copy and edit is provided in ~/source/RMS/Scripts/stations_template.csv"
    exit 1
fi

# Detect the platform. On Raspberry Pi a recommended camera limit applies:
# 1 camera, or 4 on a Pi5 or later (the tested ceiling). The limit is
# advisory and can be overridden at the prompt.
PI_MODEL_FILE="${RMS_PI_MODEL_FILE:-/sys/firmware/devicetree/base/model}"
IS_PI=0
RECOMMENDED=0   # 0 = no recommendation (generic Linux)
if [[ -f "$PI_MODEL_FILE" ]]; then
    model=$(tr -d '\0' < "$PI_MODEL_FILE")
    if [[ "$model" == *'Raspberry'* ]]; then
        IS_PI=1
        # The first number in the model name is the board generation, which
        # the keyboard models carry in the hundreds (Pi 400 = Pi4, 500 = Pi5)
        pi_num=$(printf '%s' "$model" | grep -oE '[0-9]+' | head -1)
        if [[ -n "$pi_num" && "$pi_num" -ge 100 ]]; then
            pi_num=$(( pi_num / 100 ))
        fi
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
    # The launch entries need a terminal - install lxterminal
    TERMINAL=lxterminal
    echo "Neither lxterminal nor gnome-terminal was found, installing lxterminal..."
    if ! { command -v apt-get >/dev/null 2>&1 && sudo apt-get install -y lxterminal; }; then
        echo ""
        echo "WARNING: lxterminal could not be installed. The launch entries created"
        echo "now will not start capture until lxterminal is installed."
        echo ""
    fi
fi

# Get user's desktop directory
Desktop=$(xdg-user-dir DESKTOP 2>/dev/null)
if [[ -z "$Desktop" ]]; then
    Desktop="$HOME/Desktop"
fi

DefStation=$(uppercase "$(get_config stationID ~/source/RMS/.config)")

# Stations already present (an empty ~/source/Stations does not count)
stations_before=$(count_stations)
converted=0

# A configured legacy single-camera setup is offered conversion first
if [[ $stations_before -eq 0 && -n "$DefStation" && "$DefStation" != "XX0001" ]]; then
    # the station ID names the station directories, so it must be safe as one
    if [[ ! "$DefStation" =~ ^[A-Z0-9_-]+$ ]]; then
        echo "The stationID '${DefStation}' in ~/source/RMS/.config is not a valid station ID."
        echo "Please correct it, then run add_Station again."
        exit 1
    fi
    migrate_legacy || exit 0
elif [[ -f ~/.rmsautorunflag && "$DefStation" == "XX0001" && $stations_before -eq 0 ]]; then
    echo "Please run RMS_FirstRun and configure your 1st station,"
    echo "then run add_Station again to convert it or add more cameras."
    exit
fi

# Ensure required directories exist (~/source/Stations is created with the
# first station)
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

        # reject IDs that would break directory names, and the placeholder
        # ID of an unconfigured station
        if [[ ! "$item" =~ ^[A-Za-z0-9_-]+$ || "$(uppercase "$item")" == "XX0001" ]]; then
            echo "Skipping invalid station ID from CSV: ${item}"
            continue
        fi

        create_station "$item" || continue

        config_file=~/source/Stations/${item}/.config

        # Apply any additional CSV columns that match .config keys
        for i in "${!HEADERS[@]}"; do
            header="${HEADERS[$i]}"
            value="${VALUES[$i]}"

            # Skip station_id, camera_ip (already handled), and empty values
            [[ "$header" == "station_id" || "$header" == "stationID" || "$header" == "camera_ip" ]] && continue
            [[ -z "$value" ]] && continue

            # Apply it if this header exists as a key in .config
            if set_config "${header}" "${value}" "${config_file}"; then
                echo "  Set ${header}: ${value}"
            fi
        done

        # Apply camera_ip to the device URL if the column exists - after the
        # other columns, so it also applies to a device URL from the CSV
        if [[ $camera_ip_col -ne -1 && -n "${VALUES[$camera_ip_col]}" ]]; then
            camera_ip="${VALUES[$camera_ip_col]}"
            if [[ ! "$camera_ip" =~ ^[A-Za-z0-9.-]+$ ]]; then
                echo "  Invalid camera_ip, not applied: ${camera_ip}"
            elif grep -q '^device:.*rtsp://' "${config_file}"; then
                # Replace the host of the rtsp:// URL - an IP address or a
                # hostname, after the user:password@ part if there is one
                sed -i -E "s|^(device:.*rtsp://([^/@]*@)?)[^:/@]+|\1${camera_ip}|" "${config_file}"
                echo "  Set camera IP: ${camera_ip}"
            else
                echo "  The device setting has no rtsp:// URL, camera_ip not applied"
            fi
        fi

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
        # drop buffered keypresses (e.g. the tail of an arrow key escape
        # sequence left by a single-key read) before taking a typed entry
        while read -t 0.01; do :; done
        read -r -p "Enter station ID, <cr> to end: " this_Station
        this_Station=$(uppercase "$(trim "$this_Station")")
        if [[ -z "$this_Station" ]]; then
            break
        fi
        if [[ "$this_Station" == "XX0001" ]]; then
            echo "  XX0001 is the placeholder ID of an unconfigured station, please enter your station code."
            continue
        fi
        if [[ ! "$this_Station" =~ ^[A-Z]{2}[A-Z0-9]{4}$ ]]; then
            echo "  A station code has 2 country letters followed by 4 characters, e.g. US01AB."
            # Only offer to keep a nonstandard entry when it is still safe
            # to use as a directory name and a .config value
            if [[ "$this_Station" =~ ^[A-Z0-9_-]+$ ]]; then
                read -n1 -r -p "  Use '${this_Station}' anyway? Press Y to accept it, any other key to retype: " key
                echo ""
                if [[ "$key" != "y" && "$key" != "Y" ]]; then
                    continue
                fi
            else
                continue
            fi
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

# Remove launch entries left over from the pre-merge per-platform scripts -
# their different file names would otherwise double-launch the station at
# login
for dir in ~/source/Stations/*/; do
    [[ -f "${dir}/.config" ]] || continue
    station=$(basename "$dir")
    rm -f ~/.config/autostart/"${station}_StartCap.desktop" \
          "${Desktop}/${station}_StartCap.desktop" \
          "${Desktop}/Show_LiveStream-${station}.desktop" \
          "${Desktop}/${station}-Show_LiveStream.desktop"
done

# Stations on one machine share its disk, and the daily post processing
# reboot of one would kill the captures of the others. When this run added
# a station, every station's config is updated for the new count. Values
# tuned by hand are kept as far as that allows: the reboot is only ever
# disabled, never enabled, and the reserved free space only ever raised.
# Each station's own next night is already accounted for when it frees
# space, so the reserve must cover the other cameras' nights (the FF files
# of a 720p camera take about 15 GB on a 12 hour night): 20 GB per station.
if [[ $total -gt 1 && $total -gt $stations_before ]]; then
    reserve=$(( total * 20 ))
    for dir in ~/source/Stations/*/; do
        [[ -f "${dir}/.config" ]] || continue
        set_config reboot_after_processing false "${dir}/.config"
        current=$(get_config extra_space_gb "${dir}/.config")
        if ! awk -v c="$current" -v r="$reserve" 'BEGIN { exit !(c + 0 >= r) }'; then
            set_config extra_space_gb "${reserve}" "${dir}/.config"
        fi
    done
    echo -e "\nAll ${total} station configs: reboot_after_processing disabled, extra_space_gb at least ${reserve}"
fi

# remove comment from last line of wayfire.ini to enable window cascade (Pi image)
if [[ $total -gt 1 && -f ~/.config/wayfire.ini ]]; then
    sed -i s/#mode/mode/ ~/.config/wayfire.ini
fi

# Make sure the Desktop StartCapture entry points at RMS_StartCapture.sh,
# which handles both the legacy and the multi-camera data structure (the
# entry may be a stale copy rather than a symlink on older installs).
# On the Pi images FirstRun also runs this launcher at boot.
ln -sf ~/source/RMS/Scripts/RMS_StartCapture.sh "${Desktop}/RMS_StartCapture.sh"

# Generate SSH keys if not present, in the same PEM format as RMS_FirstRun
if [[ ! -f ~/.ssh/id_rsa ]]; then
    # ssh-keygen does not create a missing ~/.ssh
    [[ -d ~/.ssh ]] || mkdir -m 700 ~/.ssh
    ssh-keygen -t rsa -m PEM -N "" -f ~/.ssh/id_rsa -q
    ln -sf ~/.ssh/id_rsa.pub "${Desktop}/id_rsa.pub"
    echo "SSH keys generated in ~/.ssh"
    echo "Your new id_rsa.pub public key file is now placed on the Desktop."
    echo "Be sure to send a copy of this file to Denis"
fi

# Activate the RMS virtual environment in new shells if not done already
if ! grep -q "source ~/vRMS/bin/activate" ~/.bashrc; then
    cat <<EOF >> ~/.bashrc

# Activate RMS
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

# Create Desktop shortcut for editing the station .config files. It runs in
# a terminal, which its prompts need (this replaces the bare Edit_configs
# script of earlier versions, which a double-click ran without one)
rm -f "${Desktop}/Edit_configs"
cat <<- EOF > "${Desktop}/Edit_configs.desktop"
	[Desktop Entry]
	Name=Edit_configs
	Type=Application
	Exec=$(term_exec "Edit_configs" "${HOME}/source/RMS/Scripts/MultiCamLinux/EditConfigs.sh")
	Hidden=false
	NoDisplay=false
	Icon=${TERMINAL}
	EOF
gio set "${Desktop}/Edit_configs.desktop" metadata::trusted true 2>/dev/null
chmod +x "${Desktop}/Edit_configs.desktop"

echo -e "\nStation configuration complete"

if [[ $(count_stations) -gt 0 ]]; then
    echo ""
    echo "The settings of each camera are in ~/source/Stations/<station_ID>/.config,"
    echo "which the Edit_configs shortcut on the Desktop opens for editing."
    if [[ $converted -eq 1 ]]; then
        echo "The config in ~/source/RMS is no longer used for capture."
    fi
fi
