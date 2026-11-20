#!/bin/bash

# UDP Buffer Size Configuration Script
# -----------------------------------
# Purpose: Configures system UDP buffer sizes for GStreamer UDP streaming
# Usage: sudo ./Scripts/UpdateBuffers.sh [--check] [--yes]
#
#   --check  Report whether the buffers need raising and exit. Requires no
#            root privileges: reading sysctl values is unprivileged.
#            Exit 0 = buffers are adequate, 10 = they need raising.
#   --yes    Apply the recommended values without prompting.
#
# This script checks and optionally updates UDP buffer sizes to handle
# GStreamer's UDP source requirements (rtspsrc udp-buffer-size, default 16MB).
# Default Linux settings are often too small, causing dropped frames.
#
# The script will:
# - Show current buffer sizes
# - Warn if below recommended values (1MB min, 16MB recommended)
# - Create backup of original settings
# - Update settings if confirmed
# - Show before/after comparison

# Configuration
RECOMMENDED_SIZE=16777216  # 16MB in bytes - must be >= rtspsrc udp-buffer-size in BufferedCapture.py
MIN_RECOMMENDED=1048576    # 1MB in bytes (old default; below this UDP RtspSrc bursts overflow)

CHECK_ONLY=false
ASSUME_YES=false

# Parse arguments before the root check, so --check works unprivileged
while [ $# -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --yes|-y)
            ASSUME_YES=true
            shift
            ;;
        --help|-h)
            sed -n '3,10p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sudo $0 [--check] [--yes]" >&2
            exit 1
            ;;
    esac
done

# Function to convert bytes to human readable format
human_readable() {
    local bytes=$1
    if [ $bytes -ge 1048576 ]; then
        echo "$(( bytes / 1048576 )) MB"
    else
        echo "$(( bytes / 1024 )) KB"
    fi
}

# Return 0 if both buffers are at or above the recommended size, 1 otherwise.
# Reads /proc/sys via sysctl, which needs no privileges.
buffers_adequate() {
    local rmem wmem
    rmem=$(sysctl -n net.core.rmem_max 2>/dev/null) || return 1
    wmem=$(sysctl -n net.core.wmem_max 2>/dev/null) || return 1

    if [ "$rmem" -lt "$RECOMMENDED_SIZE" ] || [ "$wmem" -lt "$RECOMMENDED_SIZE" ]; then
        return 1
    fi
    return 0
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

    # Report how far below the thresholds we are, if at all
    if [ $current_rmem_max -lt $MIN_RECOMMENDED ] || [ $current_wmem_max -lt $MIN_RECOMMENDED ]; then
        echo "WARNING: Current buffer sizes are below the minimum recommended size ($(human_readable $MIN_RECOMMENDED))"
        echo "This may cause issues with GStreamer UDP buffer allocation."
    elif [ $current_rmem_max -lt $RECOMMENDED_SIZE ] || [ $current_wmem_max -lt $RECOMMENDED_SIZE ]; then
        echo "NOTE: Current buffer sizes are below the recommended size ($(human_readable $RECOMMENDED_SIZE))"
        echo "GStreamer requests $(human_readable $RECOMMENDED_SIZE); the kernel clamps the request to these values."
    fi
}

# --check: report status and exit without touching anything
if [ "$CHECK_ONLY" = true ]; then
    if buffers_adequate; then
        echo "UDP buffers are at or above the recommended size ($(human_readable $RECOMMENDED_SIZE))."
        exit 0
    fi
    show_settings
    exit 10
fi

# Check if running as root (only the applying path needs privileges)
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Show initial settings and check if an update is needed
echo "CHECKING CURRENT SETTINGS:"
show_settings

if buffers_adequate; then
    echo -e "\nCurrent buffer sizes are at or above recommended values."
    exit 0
fi

# Confirm, unless --yes was given
if [ "$ASSUME_YES" != true ]; then
    if [ ! -t 0 ]; then
        # No terminal to ask on. Fail loudly rather than silently doing nothing,
        # so a calling script cannot mistake a no-op for success.
        echo "ERROR: Buffers need raising but stdin is not a terminal. Re-run with --yes." >&2
        exit 1
    fi

    echo -e "\nWould you like to update the buffer sizes to the recommended values? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "No changes made. Exiting..."
        exit 0
    fi
fi
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
# Required for GStreamer UDP streaming (rtspsrc udp-buffer-size, default 16MB)

net.core.rmem_max=$RECOMMENDED_SIZE
net.core.wmem_max=$RECOMMENDED_SIZE
EOF

echo "Created $SYSCTL_DROP_IN with:"
echo "  net.core.rmem_max=$RECOMMENDED_SIZE"
echo "  net.core.wmem_max=$RECOMMENDED_SIZE"

# Comment out any rmem_max/wmem_max lines left in /etc/sysctl.conf.
# It loads after our drop-in at boot and would otherwise override it.
SYSCTL_CONF="/etc/sysctl.conf"
if [ -f "$SYSCTL_CONF" ] && grep -Eq '^[[:space:]]*net\.core\.(rmem|wmem)_max[[:space:]]*=' "$SYSCTL_CONF"; then
    echo -e "\nFound conflicting UDP buffer settings in $SYSCTL_CONF; disabling them so the drop-in wins at boot."
    cp -n "$SYSCTL_CONF" "${SYSCTL_CONF}.rms.bak" && echo "Backed up to ${SYSCTL_CONF}.rms.bak"
    sed -i -E 's@^([[:space:]]*net\.core\.(rmem|wmem)_max[[:space:]]*=.*)@# Disabled by UpdateBuffers.sh: \1@' "$SYSCTL_CONF"
    echo "Commented out stale rmem_max/wmem_max lines in $SYSCTL_CONF"
fi

# Apply changes immediately
echo "Applying changes..."
sysctl -p "$SYSCTL_DROP_IN" >/dev/null 2>&1

echo -e "\nAFTER CHANGES:"
show_settings

# Verify the values actually took, rather than assuming
if ! buffers_adequate; then
    echo -e "\nERROR: Buffers are still below the recommended size after applying changes." >&2
    echo "Check for a conflicting setting in /etc/sysctl.d/ or a read-only /proc." >&2
    exit 1
fi

echo -e "\nDone! Settings will persist across reboots via $SYSCTL_DROP_IN"
