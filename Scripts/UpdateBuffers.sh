#!/bin/bash

# UDP Buffer Size Configuration Script
# -----------------------------------
# Purpose: Configures system UDP buffer sizes for GStreamer UDP streaming
# Usage: sudo ./Scripts/UpdateBuffers.sh [--check] [--yes]
#
#   --check   Report whether the buffers are already big enough and exit. Needs
#             no privileges and changes nothing, so a caller can ask first and
#             only reach for sudo when there is work to do.
#   --yes     Accepted for callers that expect to confirm a prompt. This script
#             is non-interactive, so it changes nothing.
#
# GStreamer's rtspsrc udp-buffer-size defaults to 16MB, but the Linux default
# net.core.rmem_max/wmem_max are much smaller, causing dropped frames.
#
# This script:
# - Writes a systemd sysctl drop-in so 16MB persists across reboots
# - Disables any stale rmem_max/wmem_max lines in /etc/sysctl.conf that would
#   otherwise override the drop-in at boot
# - Re-applies all config the way boot does, then verifies the result
#
# It is idempotent and non-interactive, so it is safe to re-run.

# Configuration
RECOMMENDED_SIZE=16777216  # 16MB in bytes - must be >= rtspsrc udp-buffer-size in BufferedCapture.py

CHECK_ONLY=false

# Parsed before the root check so --check works unprivileged
while [ $# -gt 0 ]; do
    case "$1" in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --yes|-y)
            # Nothing to confirm - kept so callers written against the
            # interactive version of this script keep working
            shift
            ;;
        --help|-h)
            sed -n '3,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sudo $0 [--check] [--yes]" >&2
            exit 1
            ;;
    esac
done

# Report only: are the live limits already at least what GStreamer asks for?
# Persistence is not checked here - re-running the script is cheap and idempotent
if [ "$CHECK_ONLY" = true ]; then
    # A kernel without these knobs (or a non-Linux host) reports nothing rather
    # than failing, so default the reading itself, not just the command
    r=$(sysctl -n net.core.rmem_max 2>/dev/null || true)
    w=$(sysctl -n net.core.wmem_max 2>/dev/null || true)
    r=${r:-0}
    w=${w:-0}
    if [ "$r" -ge "$RECOMMENDED_SIZE" ] && [ "$w" -ge "$RECOMMENDED_SIZE" ]; then
        echo "UDP buffers are already at least $RECOMMENDED_SIZE bytes (rmem_max=$r, wmem_max=$w)"
        exit 0
    fi
    echo "UDP buffers are below $RECOMMENDED_SIZE bytes (rmem_max=$r, wmem_max=$w)"
    exit 1
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Convert bytes to human readable format
human_readable() {
    local bytes=$1
    if [ "$bytes" -ge 1048576 ]; then
        echo "$(( bytes / 1048576 )) MB"
    else
        echo "$(( bytes / 1024 )) KB"
    fi
}

# Display current buffer settings
show_settings() {
    local r=$(sysctl -n net.core.rmem_max)
    local w=$(sysctl -n net.core.wmem_max)
    echo "  rmem_max: $r bytes ($(human_readable $r))"
    echo "  wmem_max: $w bytes ($(human_readable $w))"
}

echo "BEFORE:"
show_settings
echo

# 1. Write the drop-in so the setting persists across reboots.
# The 99- prefix loads it late; /etc/sysctl.d works on all systemd distros
# (Raspberry Pi OS, Ubuntu, Debian).
SYSCTL_DROP_IN="/etc/sysctl.d/99-rms-udp-buffers.conf"
[ -d /etc/sysctl.d ] || mkdir -p /etc/sysctl.d
cat > "$SYSCTL_DROP_IN" << EOF
# RMS UDP Buffer Configuration
# Created by UpdateBuffers.sh
# Required for GStreamer UDP streaming (rtspsrc udp-buffer-size, default 16MB)

net.core.rmem_max=$RECOMMENDED_SIZE
net.core.wmem_max=$RECOMMENDED_SIZE
EOF
echo "Wrote $SYSCTL_DROP_IN (rmem_max=wmem_max=$RECOMMENDED_SIZE)"

# 2. Disable any rmem_max/wmem_max lines left in /etc/sysctl.conf. It is loaded
# after our drop-in at boot (via the 99-sysctl.conf symlink), so a stale line
# there - e.g. from an older version of this script - would override the
# drop-in on every reboot. Run this unconditionally: persistence depends on the
# config files, not on the current live value.
SYSCTL_CONF="/etc/sysctl.conf"
if [ -f "$SYSCTL_CONF" ] && grep -Eq '^[[:space:]]*net\.core\.(rmem|wmem)_max[[:space:]]*=' "$SYSCTL_CONF"; then
    echo "Found conflicting settings in $SYSCTL_CONF; disabling them so the drop-in wins at boot."
    # Back up only once, preserving the true pre-edit original. Avoid 'cp -n',
    # whose behavior is non-portable and warns on newer coreutils.
    if [ ! -e "${SYSCTL_CONF}.rms.bak" ]; then
        cp "$SYSCTL_CONF" "${SYSCTL_CONF}.rms.bak" && echo "Backed up to ${SYSCTL_CONF}.rms.bak"
    fi
    sed -i -E 's@^([[:space:]]*net\.core\.(rmem|wmem)_max[[:space:]]*=.*)@# Disabled by UpdateBuffers.sh: \1@' "$SYSCTL_CONF"
fi

# 3. Re-apply all sysctl config the same way systemd does at boot, so the values
# shown below reflect what will actually survive a reboot.
echo "Applying..."
sysctl --system >/dev/null 2>&1

echo
echo "AFTER:"
show_settings
echo

# 4. Verify the effective value is what we want. If something still overrides it,
# report every file that defines it so the culprit is easy to find.
current=$(sysctl -n net.core.rmem_max)
if [ "$current" -ge "$RECOMMENDED_SIZE" ]; then
    echo "Done! net.core.rmem_max is $(human_readable $current); it will persist across reboots via $SYSCTL_DROP_IN"
else
    echo "WARNING: net.core.rmem_max is still $(human_readable $current) after applying all config."
    echo "Something sets it lower after our drop-in. Files that define it:"
    grep -rnE 'net\.core\.(rmem|wmem)_max' /etc/sysctl.conf /etc/sysctl.d/ /run/sysctl.d/ /usr/lib/sysctl.d/ /lib/sysctl.d/ 2>/dev/null
    exit 1
fi
