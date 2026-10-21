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
#   0 2 * * * /path/to/GRMSUpdater.sh --term gnome-terminal --force
# This eliminates permission issues and follows the principle of least privilege.
#
# OPTIONAL gnome-terminal PROFILE: pass --profile <name> to launch restarted
# stations with a specific gnome-terminal profile (e.g. a fixed-size profile so
# windows tile predictably). Ignored by other terminals; if the profile doesn't
# exist gnome-terminal warns and falls back to its default, so it is safe to
# pass everywhere. Example:
#   0 2 * * * /path/to/GRMSUpdater.sh --term gnome-terminal --profile StartCapture --force
#
# OPTIONAL REBOOT: pass --reboot (always) or --reboot-if-needed (only when a reboot
# is pending, e.g. after a kernel update). The capture user needs passwordless sudo
# for shutdown, or the reboot is skipped with a warning. Stock Raspberry Pi OS grants
# the default user blanket NOPASSWD sudo, so no setup is needed there. On hardened
# setups, grant just the shutdown command (replace my_username). Pin the absolute
# path that `command -v shutdown` reports on your system - sudo matches the rule
# against that path, so /sbin/shutdown vs /usr/sbin/shutdown matters:
#   echo 'my_username ALL=(ALL) NOPASSWD: /usr/sbin/shutdown' | sudo tee /etc/sudoers.d/rms-reboot > /dev/null
#   sudo chmod 0440 /etc/sudoers.d/rms-reboot

# Lock will be automatically released when script exits

# Enable strict error handling
set -Eeuo pipefail
trap 'echo "Error: Script failed at line $LINENO"' ERR

# Ensure bash is used as the shell for all child processes
# (fixes issues when script is run from cron which sets SHELL=/bin/sh)
export SHELL=/bin/bash

# Function to log messages via syslog
log_message() {
    local message="$1"
    # Log to syslog and also echo for interactive runs
    logger -t rms_updater "$message"
    echo "$message"
}

# ------------------------------------------------------------
#  regex_for() – match ONLY the StartCapture.sh argv
#  • anchors on argv-0 (script path) so terminals / python lines don't match
#  • station ID must be a standalone argument (not part of a path)
# ------------------------------------------------------------
regex_for() {
    # Match ONLY the actual bash script process, not terminals or wrappers
    # ^/bin/bash        must start with /bin/bash (the actual script interpreter)
    # [[:space:]]+      one-or-more spaces
    # .*/StartCapture\.sh  script path
    # [[:space:]]+      one-or-more spaces  
    # $1                station ID
    # ([[:space:]]|$)   end of arg or end of string
    echo '^/bin/bash[[:space:]]+.*/StartCapture\.sh[[:space:]]+'"$1"'([[:space:]]|$)'
}

# ------------------------------------------------------------
#  should_reboot() – check if system reboot is requested
# ------------------------------------------------------------
should_reboot() {
    # Reset the kernel target; only the kernel-mismatch path below sets it.
    REBOOT_KERNEL_TARGET=""

    if [[ "$REBOOT_MODE" == "always" ]]; then
        return 0
    elif [[ "$REBOOT_MODE" == "if-needed" ]]; then
        # Ubuntu/Debian: flag file created by update-notifier-common or linux-base ≥4.13.
        # This is the reliable signal: it lives on tmpfs (/run) and is cleared on reboot,
        # so it can never cause a reboot loop.
        if [[ -f /var/run/reboot-required ]]; then
            return 0
        fi
        # Fallback: compare running kernel to latest installed (works on RPi OS / Debian).
        local running latest last_target
        running="$(uname -r)"
        latest="$(ls /lib/modules/ | sort -V | tail -1)"
        if [[ -n "$latest" && "$running" != "$latest" ]]; then
            # Loop guard: if the latest installed kernel never becomes the running one
            # (unbootable kernel, bootloader pinned to an older version, RPi
            # firmware/modules mismatch), an unattended cron run would otherwise reboot
            # on every invocation. We stamp the kernel we last rebooted for (in
            # do_reboot, i.e. only when a reboot is actually issued) and refuse to
            # reboot again for the same target.
            last_target=""
            [[ -f "$REBOOT_STAMP_FILE" ]] && last_target="$(cat "$REBOOT_STAMP_FILE" 2>/dev/null || true)"
            if [[ "$last_target" == "$latest" ]]; then
                log_message "Kernel mismatch (running $running, installed $latest) but a reboot was already attempted for this kernel - skipping to avoid a reboot loop"
                return 1
            fi
            log_message "Kernel mismatch: running $running, installed $latest"
            REBOOT_KERNEL_TARGET="$latest"
            return 0
        fi
    fi
    return 1
}

# ------------------------------------------------------------
#  reboot_available() – true if passwordless `sudo shutdown` works
# ------------------------------------------------------------
reboot_available() {
    # Probe with the resolved absolute path so this matches what do_reboot runs and
    # what the sudoers rule pins (sudo matches rules against the absolute command path).
    [[ -n "${SHUTDOWN_BIN:-}" ]] && sudo -n "$SHUTDOWN_BIN" --help >/dev/null 2>&1
}

# ------------------------------------------------------------
#  do_reboot() – record the kernel loop-guard stamp (if any) and reboot.
#  Stamping here (not in should_reboot) means we only suppress future
#  reboots for kernels we actually attempted, so enabling sudo later still
#  allows the pending reboot to proceed. Returns the status of shutdown.
# ------------------------------------------------------------
do_reboot() {
    if [[ -n "${REBOOT_KERNEL_TARGET:-}" ]]; then
        echo "$REBOOT_KERNEL_TARGET" > "$REBOOT_STAMP_FILE" 2>/dev/null || true
    fi
    log_message "Rebooting system..."
    sudo -n "$SHUTDOWN_BIN" -r now
}

# ------------------------------------------------------------
#  stop_stations() – gracefully stop all running RMS stations
# ------------------------------------------------------------
stop_stations() {
    if [[ ${#RunList[@]} -gt 0 ]]; then
        log_message "Gracefully stopping ${#RunList[@]} running RMS stations: ${RunList[*]}"

        # First, try graceful shutdown with SIGTERM for each station (StartCapture forwards as SIGINT)
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
        SHUTDOWN_TIMEOUT=600  # 10 minutes - adjust based on your typical shutdown time
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
}

# Note: When run from user cron, DISPLAY may not be set. Terminal launching will fall back to tmux if needed.

# Helper function to launch terminal using preferred terminal
launch_term() {                            # $1 = title, $2… = cmd+args
    local title=$1; shift
    local cmd pid
    
    # Set up debug logging
    local LOGDIR="$USER_HOME/RMS_data/logs"
    local LOGFILE="$LOGDIR/${title}_launcher_$(date +%F_%T).log"
    mkdir -p "$LOGDIR"
    
    log_message "Launching terminal '$PREFERRED_TERM' for station $title"
    log_message "Command: $*"
    
    case "$PREFERRED_TERM" in
        lxterminal)
            # Set SHELL to /bin/bash to fix lxterminal server mode issue
            export SHELL=/bin/bash
            # one quoted string after -e:
            cmd=(lxterminal --title="$title" \
                  -e "bash -c 'export GRMS_AUTO=1; exec \"\$@\"' _ $*")
            ;;
        kitty)
            cmd=(env GRMS_AUTO=1 kitty -T "$title" "$@")
            ;;
        foot)
            cmd=(env GRMS_AUTO=1 foot --app-id="$title" -e "$@")
            ;;
        footclient)
            cmd=(env GRMS_AUTO=1 footclient --app-id="$title" -- "$@")
            ;;
        gnome-terminal)
            # build one properly-quoted payload string
            local payload
            payload=$(printf '%q ' "$@")          # quote every arg
            # optional profile (matches the --profile=... the autostart entries use)
            local profile_args=()
            [[ -n "$GTERM_PROFILE" ]] && profile_args=(--profile="$GTERM_PROFILE")
            cmd=(gnome-terminal "${profile_args[@]}" --title="$title" \
                 -- bash -lc "export GRMS_AUTO=1; exec $payload")
            ;;
        tmux)
            tmux has-session -t "$title" 2>/dev/null || \
                tmux new -d -s "$title" "export GRMS_AUTO=1; exec $*"
            return $?
            ;;
        *)
            log_message "Unknown terminal '$PREFERRED_TERM'"
            return 1
            ;;
    esac

    # Check if we need a display for this terminal type
    if [[ "$PREFERRED_TERM" != "tmux" && -z "$DISPLAY" ]]; then
        log_message "Error: No DISPLAY available for GUI terminal '$PREFERRED_TERM'"
        log_message "Try running with --term tmux or from a graphical session"
        return 1
    fi
    
    # spawn the terminal (with logging for debugging)
    log_message "Executing: ${cmd[*]}"
    ( exec 200>&-                     # Close the flock fd so children can't inherit it
      setsid "${cmd[@]}" >"$LOGFILE" 2>&1
    ) &
    local term_pid=$!
    log_message "Terminal PID: $term_pid"
    
    # wait until the real StartCapture for this station shows up (max 10s)
    local tries=0
    local pat="StartCapture\\.sh[[:space:]]+$title([[:space:]]|$)"
    log_message "Waiting for process pattern: $pat"
    
    until pgrep -f -- "$pat" >/dev/null; do
        (( tries++ > 20 )) && { 
            log_message "Terminal failed for $title after 10 seconds"
            log_message "Check log file: $LOGFILE"
            return 1
        }
        sleep 0.5
    done
    
    log_message "Station $title started successfully"
    return 0
}

# -------------------------------------------------------------------
#  restart_stations  – relaunch each station in its own lxterminal
# -------------------------------------------------------------------
restart_stations() {
    local stations_to_restart=("$@")

    if [[ ${#stations_to_restart[@]} -eq 0 ]]; then
        log_message "No stations to restart"
        return
    fi

    log_message "Restarting ${#stations_to_restart[@]} stations: ${stations_to_restart[*]}"

    for station in "${stations_to_restart[@]}"; do
        log_message "Starting station $station"
        sleep 5   # small stagger so they don't all open at once

        if ! launch_term "$station" \
              "$RMS_DIR/Scripts/MultiCamLinux/StartCapture.sh" "$station"; then
            log_message "Failed to start station $station – continuing"
        fi
    done
}

# ------------------------------------------------------------
#  usage() – print help and exit
# ------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: GRMSUpdater.sh [options]

Stops running RMS stations, updates RMS, then restarts them. With no
positional argument all configured stations (in ~/source/Stations) are
restarted; with any positional argument only the previously-running
stations are restarted.

Options:
  --term <terminal>   Terminal to launch stations in (default: lxterminal).
                      One of: lxterminal, kitty, foot, footclient,
                      gnome-terminal, tmux.
  --profile <name>    gnome-terminal profile to launch with. Ignored by
                      other terminals; gnome-terminal falls back to its
                      default profile if <name> does not exist.
  --force             Restart stations even if RMS is already up to date.
  --reboot            Always reboot after updating.
  --reboot-if-needed  Reboot only when one is pending (e.g. kernel update).
  -h, --help          Show this help and exit.
EOF
}

# Handle help before acquiring the lock or touching anything, so it works
# even when another instance is running.
for _arg in "$@"; do
    case "$_arg" in
        -h|--help) usage; exit 0 ;;
    esac
done

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
GTERM_PROFILE=""                # gnome-terminal profile to launch with (empty = terminal's default)
REBOOT_MODE="none"
REBOOT_KERNEL_TARGET=""          # kernel that should_reboot() flagged (for the loop-guard stamp)
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --term)
            PREFERRED_TERM="$2"
            shift 2
            ;;
        --profile)
            # gnome-terminal profile name (ignored by other terminals).
            # gnome-terminal warns and falls back to its default if the profile
            # doesn't exist, so passing this on a host without it is harmless.
            GTERM_PROFILE="$2"
            shift 2
            ;;
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        --reboot-if-needed)
            REBOOT_MODE="if-needed"
            shift
            ;;
        --reboot)
            REBOOT_MODE="always"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            log_message "Error: unknown option '$1'"
            usage >&2
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    set -- "${POSITIONAL_ARGS[@]}"
else
    set --  # Clear positional parameters
fi

# Set up path variables (running as capture user)
USER_HOME="$HOME"
RMS_DIR="$USER_HOME/source/RMS"
STATIONS_DIR="$USER_HOME/source/Stations"
REBOOT_STAMP_FILE="$USER_HOME/.rms_reboot_kernel"  # loop-guard: kernel we last rebooted for

# Resolve the absolute path to shutdown. sudo matches sudoers rules against the
# absolute command path, and cron's minimal PATH usually omits /usr/sbin and /sbin,
# so resolve it explicitly. Distros differ (Debian/RPi OS: /usr/sbin/shutdown; others
# /sbin/shutdown); the NOPASSWD rule must pin whichever this resolves to.
SHUTDOWN_BIN="$(command -v shutdown 2>/dev/null || true)"
if [[ -z "$SHUTDOWN_BIN" ]]; then
    for _candidate in /usr/sbin/shutdown /sbin/shutdown /bin/shutdown; do
        if [[ -x "$_candidate" ]]; then
            SHUTDOWN_BIN="$_candidate"
            break
        fi
    done
fi

# Export display environment for GUI applications (needed when running from cron)
if [[ -z ${DISPLAY:-} ]]; then
    DISPLAY=$(who | awk '/\(:[0-9]/{print $NF; exit}' | tr -d '()')
    if [[ -n "$DISPLAY" ]]; then
        export DISPLAY
        log_message "Auto-detected DISPLAY=$DISPLAY"
    else
        log_message "Warning: Could not detect DISPLAY from 'who' output"
    fi
fi

if [[ -n "$DISPLAY" ]]; then
    export XAUTHORITY="$HOME/.Xauthority"
    log_message "Using DISPLAY=$DISPLAY"
    
    # Set XDG_RUNTIME_DIR if not already set (needed for some terminals)
    if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
        uid=$(id -u)
        export XDG_RUNTIME_DIR="/run/user/$uid"
        log_message "Set XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
    fi
    
    # Set up D-Bus for gnome-terminal (when running from cron)
    # Wrap in error handling to prevent crashes
    if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
        # Get UID safely
        uid=""
        if command -v id >/dev/null 2>&1; then
            uid=$(id -u 2>/dev/null || echo "1000")
        else
            uid="1000"
        fi
        
        bus_path="/run/user/${uid}/bus"
        if [[ -e "$bus_path" ]] && [[ -S "$bus_path" ]]; then
            export DBUS_SESSION_BUS_ADDRESS="unix:path=$bus_path"
            log_message "Auto-set DBUS_SESSION_BUS_ADDRESS for gnome-terminal"
        else
            log_message "D-Bus session bus not available at $bus_path (normal in cron context)"
        fi
    else
        log_message "DBUS_SESSION_BUS_ADDRESS already set: $DBUS_SESSION_BUS_ADDRESS"
    fi
else
    log_message "No DISPLAY available - GUI terminals will fail, consider using --term tmux"
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

# Check if updates are actually needed before disrupting running processes (unless --force is used)
if [[ "$FORCE_UPDATE" != "true" ]]; then
    cd "$RMS_DIR" || { log_message "Error: RMS directory not found at $RMS_DIR"; exit 1; }

    # Get current branch and check for updates (similar to RMS_Update.sh early check)
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    if [[ "$CURRENT_BRANCH" != "unknown" ]]; then
        REMOTE_SHA=$(timeout 15s git ls-remote --quiet --heads origin "refs/heads/$CURRENT_BRANCH" | cut -f1)
        LOCAL_SHA=$(git rev-parse HEAD)
        
        # Check for modified tracked files (excluding allowed config files)
        MODIFIED_FILES=$(git diff --name-only | grep -v -E '^(\.config|camera_settings\.json)$' || true)
        
        if [[ -n "$REMOTE_SHA" && "$REMOTE_SHA" == "$LOCAL_SHA" && -z "$MODIFIED_FILES" ]]; then
            if should_reboot; then
                # Only stop the running stations once we know the reboot can actually
                # happen, otherwise we'd leave them down on the no-restart early-exit path.
                if reboot_available; then
                    log_message "No RMS update needed, but system reboot is required"
                    stop_stations
                    if do_reboot; then
                        exit 0
                    fi
                    # Reboot was available a moment ago but the shutdown call itself
                    # failed; don't leave the stations down - restart the ones we stopped.
                    log_message "WARNING: Reboot command failed after stopping stations - restarting them"
                    if [[ ${#RunList[@]} -gt 0 ]]; then
                        restart_stations "${RunList[@]}"
                    fi
                    exit 1
                else
                    log_message "WARNING: System reboot needed but sudo shutdown not available - skipping reboot"
                    log_message "Configure /etc/sudoers.d/rms-reboot to enable passwordless shutdown"
                fi
            fi
            log_message "RMS is already up to date ($CURRENT_BRANCH: $LOCAL_SHA) and no tracked file modifications - no need to restart stations"
            log_message "Use --force to restart stations anyway"
            log_message "GRMSUpdater.sh completed successfully (early exit - no updates needed)"
            exit 0
        elif [[ -n "$REMOTE_SHA" && "$REMOTE_SHA" == "$LOCAL_SHA" && -n "$MODIFIED_FILES" ]]; then
            log_message "Repository up to date but tracked files modified:"
            echo "$MODIFIED_FILES" | sed 's/^/  /' | while read -r line; do log_message "$line"; done
            log_message "Proceeding with restart to restore tracked files"
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

# Stop running stations before update
stop_stations

# Run the actual RMS update
log_message "Running RMS update..."
if "$RMS_DIR/Scripts/RMS_Update.sh" >/dev/null; then
    log_message "RMS update completed successfully"
else
    log_message "Warning: RMS update failed, but continuing to restart stations since they were already stopped"
fi

# Check if system reboot is needed after update
if should_reboot; then
    log_message "Attempting system reboot after RMS update..."
    if reboot_available && do_reboot; then
        exit 0
    else
        log_message "WARNING: Reboot failed (sudo shutdown not available) - restarting stations instead"
        log_message "Configure /etc/sudoers.d/rms-reboot to enable passwordless shutdown"
    fi
fi

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

