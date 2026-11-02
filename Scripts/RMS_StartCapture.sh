#!/bin/bash

# Starts capture on either supported data structure:
# - multi-camera (~/source/Stations has station directories): every
#   configured station is started in its own terminal, staggered so
#   concurrent first-start updates cannot collide
# - legacy: the single camera is started in this terminal

echo "Starting RMS..."
sleep 10

dircount=$(find ~/source/Stations -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

if [[ $dircount -gt 0 ]]; then

    # On a single camera system concurrent first-start updates are no
    # risk, so the long stagger delay is skipped
    if [[ $dircount -le 1 ]]; then
        seconds=2
    else
        seconds=70
    fi

    # Each station runs in its own terminal window, which needs a desktop
    # session (GRMSUpdater.sh --term tmux runs the stations without one)
    if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        echo " No display to open the station windows on (e.g. over ssh), no station was started"
        exit 1
    fi

    # Detect the terminal emulator (add_Station.sh installs lxterminal on
    # desktops that have neither)
    if [[ "${XDG_CURRENT_DESKTOP:-}" == *GNOME* ]] && command -v gnome-terminal >/dev/null 2>&1; then
        TERMINAL=gnome-terminal
    elif command -v lxterminal >/dev/null 2>&1; then
        TERMINAL=lxterminal
    elif command -v gnome-terminal >/dev/null 2>&1; then
        TERMINAL=gnome-terminal
    else
        echo " Neither lxterminal nor gnome-terminal was found, no station was started"
        exit 1
    fi

    # Each station runs with its own .config, so arguments for a single
    # capture do not apply
    if [[ $# -gt 0 ]]; then
        echo " Ignoring the arguments '$*', they only apply to the single-camera data structure"
    fi

    echo " Starting all configured stations..."

    loop=0
    for Dir in ~/source/Stations/*/
      do
        Station=$(basename "$Dir")
        station_cmd="$HOME/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station}"
        echo " Starting camera ${Station}"
        case "$TERMINAL" in
            gnome-terminal)
                gnome-terminal --profile=StartCapture --title="${Station}" -- bash -c "${station_cmd}" &
                ;;
            lxterminal)
                lxterminal --title="${Station}" -e "${station_cmd}" &
                ;;
        esac
        echo "  waiting $seconds seconds..."
        sleep ${seconds}
        if [[ $loop = 0 ]] ; then
            seconds=10
        fi
        let loop++
      done
    echo " All cameras started"

else

    # Legacy single-camera data structure
    source ~/vRMS/bin/activate
    cd ~/source/RMS

    echo ""
    echo ""
    echo "If you need to update the RMS config file, you can do it now."
    echo "Any changes to the config file will be read only after this script is started again or the Pi is rebooted."
    echo ""
    sleep 5

    python -m RMS.StartCapture "$@"

    read -p "Press any key to continue... "

    $SHELL
fi
