#!/bin/bash

# This software is part of the Linux port of RMS
# Copyright (C) 2023  Ed Harman
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Version 2.0 major refactoring: added early update check, graceful shutdown, strict error handling, 
# user-mode execution, comprehensive terminal support, and regex helper function
#
# Version 1.9 changed parsing of the current username and display number to handle various display managers
#
# Version 1.8 changed parsing of the current username and display number to handle username ending in a digit
#
# Version 1.7 changed parsing of the current username and display number to be agnostic to either the display manager or the session manager
#
# Version 1.6 bug fixes -
# fixed this script failing when run under cron
# changed behaviour to allow for consistent run-line parsing irrespective of how a capture was initiated
# be it autostart, desktop icon or this script itself
#
# Version 1.5 fixed bugs introduced by version 1.4 changes.
#
# Version 1.5 fixed untested issue introduced by 1.4
#
# Version 1.4 fixed path issue in Run/Pid list variables
#
# Version 1.3 moved codebase into RMS/Scripts/MultiCamLinux
#
# Version 1.2 numerous fixes -
# added support for a delayed start
# fixed bug  parsing the running processes, used when called with an argument
# Version 1.1
# Changes: Fixed a bug whereby the list of running RMS processes was incorrect when the script was called with 
# an argument and the script was invoked by cron (i.e. root)
#
# read the  PID's of all RMS processes into an array and then read the number of running stations into 
# array RunList so that after killing the instances we can then update RMS and then restart the stations.
# Default behaviour if called with no arguments, - capture all the running RMS processes, kill them, update RMS, then start 
# all that are configured within directory ~/source/Stations -
#
# NOTE: This script should be run as the capture user, not root. Add to user's crontab:
#   crontab -e
#   0 2 * * * /path/to/GRMSUpdater.sh
# This eliminates permission issues and follows the principle of least privilege.

# Lock will be automatically released when script exits

# Enable strict error handling
set -Eeuo pipefail
trap 'echo "Error: Script failed at line $LINENO"' ERR

# Function to log messages via syslog
log_message() {
    local message="$1"
    # Log to syslog and also echo for interactive runs
    logger -t rms_updater "$message"
    echo "$message"
}

# Function to generate regex pattern for station matching
regex_for() {
    echo "/StartCapture\.sh[[:space:]]+$1([[:space:]]|$)"
}

# Log script start
log_message "GRMSUpdater.sh started with args: $*"

# Use flock to prevent multiple instances from running simultaneously
LOCKFILE="/tmp/rms_grms_updater.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    log_message "Another GRMSUpdater instance is already running. Exiting."
    exit 1
fi

# Parse command line arguments
FORCE_UPDATE=false
PREFERRED_TERM="lxterminal"     # default terminal
LAUNCH_WAIT=2                   # seconds to wait for terminal to start
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --term)
            PREFERRED_TERM="$2"
            shift 2
            ;;
        --wait)
            LAUNCH_WAIT="$2"
            shift 2
            ;;
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Set up path variables (running as capture user)
USER_HOME="$HOME"
RMS_DIR="$USER_HOME/source/RMS"
STATIONS_DIR="$USER_HOME/source/Stations"
DESKTOP_DIR="$USER_HOME/Desktop"

# Export display environment for GUI applications (needed when running from cron)
if [[ -z ${DISPLAY:-} ]]; then
    DISPLAY=$(who | awk '/\(:[0-9]/ {sub(/[()]/,"",$NF); print $NF; exit}')
    export DISPLAY
fi
export XAUTHORITY="$HOME/.Xauthority"

# Check if updates are actually needed before disrupting running processes (unless --force is used)
if [[ "$FORCE_UPDATE" != "true" ]]; then
    cd "$RMS_DIR" || { log_message "Error: RMS directory not found at $RMS_DIR"; exit 1; }

    # Get current branch and check for updates (similar to RMS_Update.sh early check)
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    if [[ "$CURRENT_BRANCH" != "unknown" ]]; then
        REMOTE_SHA=$(timeout 15s git ls-remote --quiet --heads origin "refs/heads/$CURRENT_BRANCH" | cut -f1)
        LOCAL_SHA=$(git rev-parse HEAD)
        
        if [[ -n "$REMOTE_SHA" && "$REMOTE_SHA" == "$LOCAL_SHA" ]]; then
            log_message "RMS is already up to date ($CURRENT_BRANCH: $LOCAL_SHA) - no need to restart stations"
            log_message "Use --force to restart stations anyway"
            log_message "GRMSUpdater.sh completed successfully (early exit - no updates needed)"
            exit 0
        else
            log_message "Updates available for RMS ($CURRENT_BRANCH: $LOCAL_SHA → $REMOTE_SHA) - proceeding with restart"
        fi
    else
        log_message "Warning: Could not determine current branch, proceeding with update"
    fi
else
    log_message "Force update requested - proceeding with restart regardless of update status"
    cd "$RMS_DIR" || { log_message "Error: RMS directory not found at $RMS_DIR"; exit 1; }
fi

# Find running stations by looking for StartCapture.sh processes
mapfile -t RunList < <(
    pgrep -f "Scripts/MultiCamLinux/StartCapture.sh" | while read -r pid; do
        cmdline=$(ps -p "$pid" -o args= 2>/dev/null || continue)
        if [[ "$cmdline" =~ Scripts/MultiCamLinux/StartCapture\.sh[[:space:]]+([[:alnum:]]{6}) ]]; then
            echo "${BASH_REMATCH[1]}"
        fi
    done | sort -u
)

# Only stop processes if we actually need to update
if [[ ${#RunList[@]} -gt 0 ]]; then
    log_message "Gracefully stopping ${#RunList[@]} running RMS stations for update: ${RunList[*]}"
    
    # First, try graceful shutdown with SIGTERM for each station (including all child processes)
    for station in "${RunList[@]}"; do
        log_message "Sending SIGTERM to all processes for station $station..."
        pattern=$(regex_for "$station")
        if pkill -f -TERM -- "$pattern" 2>/dev/null; then
            log_message "Sent SIGTERM to station $station processes"
        else
            log_message "Warning: No processes found for station $station (may have already exited)"
        fi
    done
    
    # Wait for processes to shut down gracefully (with reasonable timeout)
    SHUTDOWN_TIMEOUT=3600  # 1 hour - adjust based on your typical shutdown time
    WAIT_INTERVAL=5
    elapsed=0
    
    log_message "Waiting up to ${SHUTDOWN_TIMEOUT} seconds for graceful shutdown..."
    
    while [[ $elapsed -lt $SHUTDOWN_TIMEOUT ]]; do
        # Check if any station processes are still running
        still_running=()
        for station in "${RunList[@]}"; do
            pattern=$(regex_for "$station")
            if pgrep -f -- "$pattern" >/dev/null 2>&1; then
                still_running+=("$station")
            fi
        done
        
        if [[ ${#still_running[@]} -eq 0 ]]; then
            log_message "All station processes shut down gracefully after ${elapsed} seconds"
            break
        fi
        
        log_message "Still waiting for ${#still_running[@]} stations to shutdown: ${still_running[*]} (${elapsed}s elapsed)"
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
    
    # Force kill any remaining processes if timeout reached
    final_check=()
    for station in "${RunList[@]}"; do
        pattern=$(regex_for "$station")
        if pgrep -f -- "$pattern" >/dev/null 2>&1; then
            final_check+=("$station")
        fi
    done
    
    if [[ ${#final_check[@]} -gt 0 ]]; then
        log_message "Timeout reached. Force killing ${#final_check[@]} remaining stations: ${final_check[*]}"
        for station in "${final_check[@]}"; do
            log_message "Force killing all processes for station $station..."
            pattern=$(regex_for "$station")
            if pkill -f -KILL -- "$pattern" 2>/dev/null; then
                log_message "Force killed station $station processes"
            else
                log_message "Warning: Could not kill processes for station $station (may have already exited)"
            fi
        done
        
        # Give a moment for force kills to take effect
        sleep 2
    fi
    
else
    log_message "No running RMS stations found"
fi

# Note: When run from user cron, DISPLAY may not be set. Terminal launching will fall back to tmux if needed.

# Helper function to launch terminal using preferred terminal
launch_term() {                            # $1 = title, $2… = cmd+args
    local title=$1; shift
    local cmd pid

    case "$PREFERRED_TERM" in
        lxterminal)
            cmd=(lxterminal --title="$title" -e "$@")
            ;;
        kitty)
            cmd=(kitty -T "$title" "$@")
            ;;
        foot)
            cmd=(foot --app-id="$title" -e "$@")
            ;;
        footclient)
            cmd=(footclient --app-id="$title" -- "$@")
            ;;
        gnome-terminal)
            cmd=(gnome-terminal --title="$title" -- bash -lc "exec \"\$@\"" _ "$@")
            ;;
        tmux)
            tmux has-session -t "$title" 2>/dev/null || tmux new -d -s "$title" "$@"
            return $?
            ;;
        *)
            log_message "Unknown terminal '$PREFERRED_TERM'"
            return 1
            ;;
    esac

    # spawn and verify it's still alive after specified wait time
    (setsid "${cmd[@]}" >/dev/null 2>&1) & 
    pid=$!
    sleep "$LAUNCH_WAIT"
    kill -0 "$pid" 2>/dev/null
}

# Function to restart stations
restart_stations() {
    local stations_to_restart=("$@")
    
    if [[ ${#stations_to_restart[@]} -eq 0 ]]; then
        log_message "No stations to restart"
        return
    fi
    
    log_message "Restarting ${#stations_to_restart[@]} stations: ${stations_to_restart[*]}"
    
    for station in "${stations_to_restart[@]}"; do
        # Get the delay parameter from this station's desktop link (if it exists)
        local delay=""
        local desktop_file="$DESKTOP_DIR/${station}_StartCap.desktop"
        
        if [[ -f "$desktop_file" ]]; then
            # Look for explicit --delay= flag in the Exec line
            local exec_line
            exec_line=$(grep "^Exec=" "$desktop_file" 2>/dev/null || echo "")
            if [[ -n "$exec_line" ]]; then
                # Extract delay value from --delay=VALUE pattern
                if [[ "$exec_line" =~ --delay=([0-9]+) ]]; then
                    delay="${BASH_REMATCH[1]}"
                fi
            fi
        fi
        
        log_message "Starting station $station$([ -n "$delay" ] && echo " with delay $delay")"
        sleep 5
        
        if ! launch_term "$station" "$RMS_DIR/Scripts/MultiCamLinux/StartCapture.sh" "$station" "${delay:-}"; then
            log_message "Failed to start station $station - continuing with next station"
            continue
        fi
    done
}


# Run the actual RMS update
log_message "Running RMS update..."
if "$RMS_DIR/Scripts/RMS_Update.sh" >/dev/null; then
    log_message "RMS update completed successfully"
else
    log_message "Warning: RMS update failed, but continuing to restart stations since they were already stopped"
fi
sleep 10

if [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]]; then
    # Called with no args - restart all configured stations
    log_message "Will restart all configured stations post-update"
    
    # Build array of all configured stations
    mapfile -t configured_stations < <(
        for dir in "$STATIONS_DIR"/*; do
            if [[ -d "$dir" && "${dir##*/}" != "Scripts" ]]; then
                echo "${dir##*/}"
            fi
        done
    )
    
    restart_stations "${configured_stations[@]}"
    
else
    # Called with argument - only restart stations that were actually running
    log_message "Will restart only previously running stations: ${RunList[*]}"
    restart_stations "${RunList[@]}"
fi

# Log script completion
log_message "GRMSUpdater.sh completed successfully"

