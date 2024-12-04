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

# Backup original sysctl.conf
echo "Creating backup of current sysctl.conf..."
cp /etc/sysctl.conf /etc/sysctl.conf.backup.$(date +%Y%m%d_%H%M%S)

# Function to add or update sysctl setting
update_sysctl() {
    local key=$1
    local value=$2
    
    # Check if setting already exists
    if grep -q "^${key}[[:space:]]*=" /etc/sysctl.conf; then
        # Update existing setting
        sed -i "s|^${key}[[:space:]]*=.*|${key}=${value}|" /etc/sysctl.conf
        echo "Updated ${key} to ${value}"
    else
        # Add new setting
        echo "${key}=${value}" >> /etc/sysctl.conf
        echo "Added ${key}=${value}"
    fi
}

echo "Updating buffer size settings..."

# Update the settings
update_sysctl "net.core.rmem_max" "$RECOMMENDED_SIZE"
update_sysctl "net.core.wmem_max" "$RECOMMENDED_SIZE"

# Apply changes
echo "Applying changes..."
sysctl -p >/dev/null 2>&1  # Suppress detailed output

echo -e "\nAFTER CHANGES:"
show_settings

echo -e "\nDone! A backup of your original sysctl.conf has been created."