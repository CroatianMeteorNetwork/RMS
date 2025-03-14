#!/bin/bash

# Script to toggle cache mode between "write back" and "write through" and make it persistent

# Note:
# - "Write back" caching is recommended for flash storage devices like SD cards as it can improve performance
#   by reducing the number of write operations.
# - However, "write back" caching carries a risk of data loss in the event of a power failure or unexpected 
#   shutdown, as data may remain in the cache and not be written to the device immediately.
# - If data integrity is critical, consider using "write through" caching instead, which writes data directly
#   to the device without delay.

# Usage: sudo ./Scripts/ToggleCacheMode.sh

CACHE_FILE_BASE="/sys/block"
RC_LOCAL="/etc/rc.local"

# Check for sudo/root privileges
if [[ $EUID -ne 0 ]]; then
    echo "Please run this script as root or using sudo."
    exit 1
fi

# Identify storage devices
echo "Checking available storage devices..."
DEVICES=($(lsblk -nd --output NAME,TYPE | grep -w "disk" | awk '{print $1}'))

if [[ ${#DEVICES[@]} -eq 0 ]]; then
    echo "No storage devices found. Exiting."
    exit 1
fi

echo "Available devices:"
for i in "${!DEVICES[@]}"; do
    echo "$((i + 1)). ${DEVICES[i]}"
done

# Prompt the user to select a device
echo
read -p "Enter the number corresponding to your device: " DEVICE_CHOICE
if [[ ! "$DEVICE_CHOICE" =~ ^[0-9]+$ ]] || [[ "$DEVICE_CHOICE" -lt 1 ]] || [[ "$DEVICE_CHOICE" -gt "${#DEVICES[@]}" ]]; then
    echo "Invalid choice. Exiting."
    exit 1
fi

DEVICE=${DEVICES[$((DEVICE_CHOICE - 1))]}
CACHE_FILE="$CACHE_FILE_BASE/$DEVICE/queue/write_cache"

# Validate the selected device
if [[ ! -e $CACHE_FILE ]]; then
    echo "Error: Device $DEVICE does not support write cache configuration."
    exit 1
fi

# Provide information about cache modes
echo
echo "About Cache Modes:"
echo "- 'Write back':"
echo "  * Recommended for flash storage devices like SD cards."
echo "  * Improves performance by reducing the number of write operations."
echo "  * Helps extend the lifespan of flash-based storage by minimizing write amplification."
echo "  * WARNING: Increases the risk of data loss during power failures, as data may remain in the cache."
echo
echo "- 'Write through':"
echo "  * Ensures data integrity by writing immediately to storage."
echo "  * Safer for critical data but may reduce performance."
echo
echo "Please select a cache mode based on your use case."
echo

# Prompt the user to select cache mode
echo
echo "Select cache mode:"
echo "1. Write back"
echo "2. Write through"
read -p "Enter your choice [1-2]: " MODE_CHOICE

case "$MODE_CHOICE" in
    1)
        MODE="write back"
        ;;
    2)
        MODE="write through"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Apply the cache mode using `tee`
echo "$MODE" | sudo tee "$CACHE_FILE"
if [[ $? -eq 0 ]]; then
    echo "Cache mode set to '$MODE' for $DEVICE."
else
    echo "Failed to set cache mode for $DEVICE."
    exit 1
fi

# Make the change persistent
if [[ -w $RC_LOCAL ]]; then
    echo "Configuring $RC_LOCAL for persistence..."
    
    # Add or update the command in /etc/rc.local
    sed -i '/write_cache/d' $RC_LOCAL 2>/dev/null
    sed -i '/exit 0/d' $RC_LOCAL
    echo "echo \"$MODE\" | sudo tee \"$CACHE_FILE\"" >> $RC_LOCAL
    echo "exit 0" >> $RC_LOCAL
    chmod +x $RC_LOCAL

    # Enable the rc-local service
    systemctl enable rc-local
    systemctl start rc-local

    echo "Persistence added to $RC_LOCAL and service enabled."
else
    echo "Error: Cannot write to $RC_LOCAL. Check permissions."
    exit 1
fi

# Final verification
echo "Verifying persistence configuration..."
if systemctl is-enabled rc-local >/dev/null 2>&1; then
    echo "rc-local service is enabled."
else
    echo "Failed to enable rc-local service. Check systemctl logs for more details."
    exit 1
fi

echo "Setup complete. Reboot the system to test persistence."