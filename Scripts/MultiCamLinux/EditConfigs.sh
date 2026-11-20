#!/bin/bash

# Opens the .config file of each station under ~/source/Stations for
# editing, one by one. Run by the Edit_configs Desktop shortcut that
# add_Station.sh creates.

shopt -s nullglob
dirs=(~/source/Stations/*/)

if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "No stations found in ~/source/Stations"
    read -n1 -r -p 'Press any key to close... '
    exit 1
fi

editor=""
for e in mousepad leafpad gedit gnome-text-editor nano; do
    if command -v "$e" >/dev/null 2>&1; then
        editor="$e"
        break
    fi
done

if [[ -z "$editor" ]]; then
    echo "No text editor found, please install one of: mousepad leafpad gedit gnome-text-editor nano"
    read -n1 -r -p 'Press any key to close... '
    exit 1
fi

echo ""
echo "One by one, each config file under ~/source/Stations will open for editing..."
read -n1 -r -p 'Press ENTER to proceed, any other key to quit: ' key
if [[ "${key}" != "" ]]; then
    exit
fi

for Dir in "${dirs[@]}"
do
    Station=$(basename "${Dir}")
    echo ""
    echo "Editing .config file for ${Station}"
    read -n1 -r -p 'Press ENTER to edit, any other key to skip: ' key
    if [[ "${key}" = "" ]]; then
        "$editor" "${Dir}.config"
    fi
done

echo ""
echo "Done editing .config files"
sleep 2
