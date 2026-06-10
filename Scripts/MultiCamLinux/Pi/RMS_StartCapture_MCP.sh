#!/bin/bash

# Start capture on all configured stations, staggered so concurrent
# first-start updates cannot collide. On a single camera system there is
# no such risk, so the long delay is skipped

dircount=$(find ~/source/Stations -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
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

echo " Starting all configured stations post-update..."

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
