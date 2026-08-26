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

    # Detect the terminal emulator
    if [[ "${XDG_CURRENT_DESKTOP:-}" == *GNOME* ]] && command -v gnome-terminal >/dev/null 2>&1; then
        TERMINAL=gnome-terminal
    elif command -v lxterminal >/dev/null 2>&1; then
        TERMINAL=lxterminal
    else
        TERMINAL=gnome-terminal
    fi

    echo " Starting all configured stations..."

    loop=0
    for Dir in ~/source/Stations/*/
      do
        Station=$(basename "$Dir")
        echo " Starting camera ${Station}"
        if [[ "$TERMINAL" == "gnome-terminal" ]]; then
            gnome-terminal --profile=StartCapture --title=${Station} -- bash -c "$HOME/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station}" &
        else
            lxterminal --title=${Station} -e "$HOME/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station}" &
        fi
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

    mkdir -p ~/RMS_data/logs/

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
