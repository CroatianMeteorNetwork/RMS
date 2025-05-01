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

printf "%s\n" "$MODE" > "$CACHE_FILE"
if [[ $? -eq 0 ]]; then
    echo "Cache mode set to '$MODE' for $DEVICE."
else
    echo "Failed to set cache mode for $DEVICE."
    exit 1
fi

# Make the change persistent using a systemd oneshot service
SERVICE_PATH="/etc/systemd/system/write-cache@${DEVICE}.service"

cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=Set write-cache mode for %i
ConditionPathExists=$CACHE_FILE
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'printf "%s\n" "$MODE" > $CACHE_FILE'

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "write-cache@${DEVICE}.service"

# Final verification
echo "Verifying persistence configuration (systemd)..."
if systemctl is-enabled "write-cache@${DEVICE}.service" >/dev/null 2>&1; then
    echo "write-cache service is enabled."
else
    echo "Failed to enable write-cache service. Check systemctl logs for more details."
    exit 1
fi

echo "Setup complete. Reboot the system to test persistence."