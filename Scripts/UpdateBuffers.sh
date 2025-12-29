#!/bin/bash

# UDP Buffer Size Configuration Script
# -----------------------------------
# Purpose: Configures system UDP buffer sizes for GStreamer UDP streaming
# Usage: sudo ./Scripts/UpdateBuffers.sh
#
# This script checks and optionally updates UDP buffer sizes to handle
# GStreamer's UDP source requirements (512KB per camera). Default Linux
# settings are often too small, causing GStreamer warnings.
#
# The script will:
# - Show current buffer sizes
# - Warn if below recommended values (512KB min, 1MB recommended)
# - Create backup of original settings
# - Update settings if confirmed
# - Show before/after comparison


# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

# Configuration
RECOMMENDED_SIZE=1048576  # 1MB in bytes
MIN_RECOMMENDED=524288    # 512KB in bytes (GStreamer requested size)

# Function to convert bytes to human readable format
human_readable() {
    local bytes=$1
    if [ $bytes -ge 1048576 ]; then
        echo "$(( bytes / 1048576 )) MB"
    else
        echo "$(( bytes / 1024 )) KB"
    fi
}

# Function to display buffer settings
show_settings() {
    local current_rmem_max=$(sysctl -n net.core.rmem_max)
    local current_wmem_max=$(sysctl -n net.core.wmem_max)

    echo "Current buffer settings:"
    echo "----------------------"
    echo "Receive buffer max (rmem_max): $current_rmem_max bytes ($(human_readable $current_rmem_max))"
    echo "Send buffer max (wmem_max): $current_wmem_max bytes ($(human_readable $current_wmem_max))"
    echo "----------------------"

    # Check if buffers are below recommended size
    local update_needed=false
    if [ $current_rmem_max -lt $MIN_RECOMMENDED ] || [ $current_wmem_max -lt $MIN_RECOMMENDED ]; then
        echo -e "WARNING: Current buffer sizes are below the minimum recommended size (512KB)"
        echo "This may cause issues with GStreamer UDP buffer allocation."
        update_needed=true
    elif [ $current_rmem_max -lt $RECOMMENDED_SIZE ] || [ $current_wmem_max -lt $RECOMMENDED_SIZE ]; then
        echo -e "NOTE: Current buffer sizes are below the recommended size (1MB)"
        echo "Increasing them would provide more headroom for UDP operations."
        update_needed=true
    fi

    if [ "$update_needed" = true ]; then
        echo -e "\nWould you like to update the buffer sizes to the recommended values? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            return 0  # proceed with update
        else
            echo "No changes made. Exiting..."
            exit 0
        fi
    else
        echo -e "Current buffer sizes are at or above recommended values."
        exit 0
    fi
}

# Show initial settings and check if update is needed
echo "CHECKING CURRENT SETTINGS:"
show_settings
echo

# Use drop-in file in /etc/sysctl.d/ for systemd compatibility
# The 99- prefix ensures this loads last and won't be overridden
# This method works on all modern Linux distributions (Debian 7+, Ubuntu 12.04+, all Raspberry Pi OS)
SYSCTL_DROP_IN="/etc/sysctl.d/99-rms-udp-buffers.conf"

# Ensure the directory exists (should always exist on supported systems)
if [ ! -d /etc/sysctl.d ]; then
    echo "Warning: /etc/sysctl.d not found, creating it..."
    mkdir -p /etc/sysctl.d
fi

echo "Creating sysctl drop-in file: $SYSCTL_DROP_IN"

# Write the drop-in configuration file
cat > "$SYSCTL_DROP_IN" << EOF
# RMS UDP Buffer Configuration
# Created by UpdateBuffers.sh on $(date)
# Required for GStreamer UDP streaming (512KB min per camera)

net.core.rmem_max=$RECOMMENDED_SIZE
net.core.wmem_max=$RECOMMENDED_SIZE
EOF

echo "Created $SYSCTL_DROP_IN with:"
echo "  net.core.rmem_max=$RECOMMENDED_SIZE"
echo "  net.core.wmem_max=$RECOMMENDED_SIZE"

# Apply changes immediately
echo "Applying changes..."
sysctl -p "$SYSCTL_DROP_IN" >/dev/null 2>&1

echo -e "\nAFTER CHANGES:"
show_settings

echo -e "\nDone! Settings will persist across reboots via $SYSCTL_DROP_IN"