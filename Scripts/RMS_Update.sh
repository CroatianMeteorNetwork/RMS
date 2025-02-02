#!/bin/bash

# This script updates the RMS code from GitHub.
# Includes error handling, retries, and ensures critical files are never lost.

# Branch configuration (can be overridden by environment variable)
: "${RMS_BRANCH:=master}"  # Default to master if not set

# Directories, files, and variables
RMSSOURCEDIR=~/source/RMS
RMSBACKUPDIR=~/.rms_backup
CURRENT_CONFIG="$RMSSOURCEDIR/.config"
CURRENT_MASK="$RMSSOURCEDIR/mask.bmp"
BACKUP_CONFIG="$RMSBACKUPDIR/.config"
BACKUP_MASK="$RMSBACKUPDIR/mask.bmp"
SYSTEM_PACKAGES="$RMSSOURCEDIR/system_packages.txt"
UPDATEINPROGRESSFILE=$RMSBACKUPDIR/update_in_progress
LOCKFILE="/tmp/update.lock"
MIN_SPACE_MB=200  # Minimum required space in MB
RETRY_LIMIT=3
GIT_RETRY_LIMIT=5
GIT_RETRY_DELAY=60  # Seconds between git operation retries

# Function to clean up and release the lock on exit
cleanup() {
    rm -f "$LOCKFILE"
}

# Function to check available disk space
check_disk_space() {
    local dir=$1
    local required_mb=$2
    
    # Get available space in MB
    local available_mb=$(df -m "$dir" | awk 'NR==2 {print $4}')
    
    if [ "$available_mb" -lt "$required_mb" ]; then
        echo "Error: Insufficient disk space in $dir. Need ${required_mb}MB, have ${available_mb}MB"
        return 1
    fi
    return 0
}

# Retry mechanism for critical file operations
retry_cp() {
    local src=$1
    local dest=$2
    local temp_dest="${dest}.tmp"
    local retries=0

    while [ $retries -lt $RETRY_LIMIT ]; do
        if cp "$src" "$temp_dest"; then
            # Validate the copied file
            if diff "$src" "$temp_dest" > /dev/null; then
                mv "$temp_dest" "$dest"
                return 0
            else
                echo "Error: Validation failed. Retrying..."
                rm -f "$temp_dest"
            fi
        else
            echo "Error: Copy failed. Retrying..."
            rm -f "$temp_dest"
        fi
        retries=$((retries + 1))
        sleep 1
    done

    echo "Critical Error: Failed to copy $src to $dest after $RETRY_LIMIT retries."
    return 1
}

# Backup files
backup_files() {
    echo "Backing up original files..."

    # Backup .config
    if [ -f "$CURRENT_CONFIG" ]; then
        if ! retry_cp "$CURRENT_CONFIG" "$BACKUP_CONFIG"; then
            echo "Critical Error: Could not back up .config file. Aborting."
            exit 1
        fi
    else
        echo "No original .config found. Generic config will be used."
    fi

    # Backup mask.bmp
    if [ -f "$CURRENT_MASK" ]; then
        if ! retry_cp "$CURRENT_MASK" "$BACKUP_MASK"; then
            echo "Critical Error: Could not back up mask.bmp file. Aborting."
            exit 1
        fi
    else
        echo "No original mask.bmp found. Blank mask will be used."
    fi
}

# Restore files
restore_files() {
    echo "Restoring configuration and mask files..."

    # Restore .config
    if [ -f "$BACKUP_CONFIG" ]; then
        if ! retry_cp "$BACKUP_CONFIG" "$CURRENT_CONFIG"; then
            echo "Critical Error: Failed to restore .config. Aborting."
            exit 1
        fi
    else
        echo "No backup .config found - a new one will be created by the installation."
    fi

    # Restore mask.bmp
    if [ -f "$BACKUP_MASK" ]; then
        if ! retry_cp "$BACKUP_MASK" "$CURRENT_MASK"; then
            echo "Critical Error: Failed to restore mask.bmp. Aborting."
            exit 1
        fi
    else
        echo "No backup mask.bmp found - a new blank mask will be created by the installation."
    fi
}

# Function for reliable git operations
git_with_retry() {
    local cmd=$1
    local branch=$2
    local attempt=1
    
    while [ $attempt -le $GIT_RETRY_LIMIT ]; do
        echo "Attempting git $cmd (try $attempt of $GIT_RETRY_LIMIT)..."
        
        case $cmd in
            "fetch")
                if git fetch --all --prune --force --verbose; then
                    return 0
                fi
                ;;
            "reset")
                if git reset --hard "origin/$branch"; then
                    return 0
                fi
                ;;
            *)
                echo "Unknown git command: $cmd"
                return 1
                ;;
        esac
        
        echo "Git $cmd failed, waiting ${GIT_RETRY_DELAY}s before retry..."
        sleep $GIT_RETRY_DELAY
        attempt=$((attempt + 1))
    done
    
    echo "Error: Git $cmd failed after $GIT_RETRY_LIMIT attempts"
    return 1
}

# Install missing dependencies
install_missing_dependencies() {
    if [ ! -f "$SYSTEM_PACKAGES" ]; then
        echo "Warning: System packages file not found: $SYSTEM_PACKAGES"
        return
    fi

    local missing_packages=()

    # Identify missing packages
    while read -r pkg; do
        # Skip blank lines or commented lines
        [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
        
        if ! dpkg -s "$pkg" &>/dev/null; then
            missing_packages+=("$pkg")
        fi
    done < $SYSTEM_PACKAGES

    # If no missing packages, inform and return
    if [ ${#missing_packages[@]} -eq 0 ]; then
        echo "All required packages are already installed."
        return
    fi

    echo "The following packages are missing and will be installed: ${missing_packages[*]}"

    if sudo -n true 2>/dev/null; then
        echo "Passwordless sudo available. Installing missing packages..."
        sudo apt-get update
        for package in "${missing_packages[@]}"; do
            if ! sudo apt-get install -y "$package"; then
                echo "Failed to install $package. Please install it manually."
            fi
        done
    else
        # Clear screen and show prominent message for sudo
        tput clear  # Clear screen
        tput bold; tput setaf 3  # Bold yellow
        echo "
        ==============================================
        Sudo access needed for package installation
        ==============================================
        "
        tput sgr0  # Reset formatting
        
        sudo apt-get update
        for package in "${missing_packages[@]}"; do
            if ! sudo apt-get install -y "$package"; then
                echo "Failed to install $package. Please install it manually."
            fi
        done
    fi
}

main() {
    # Check for running instance FIRST
    if [ -f "$LOCKFILE" ]; then
        LOCK_PID=$(cat "$LOCKFILE")
        if ps -p "$LOCK_PID" > /dev/null 2>&1; then
            echo "Another instance of the script is already running. Exiting."
            exit 1
        else
            echo "Stale lock file found. Removing it and continuing."
            rm -f "$LOCKFILE"
        fi
    fi

    # Create lock file immediately
    echo $$ > "$LOCKFILE"
    trap cleanup EXIT

    # Run space check before anything else
    echo "Checking available disk space..."
    check_disk_space "$RMSSOURCEDIR" "$MIN_SPACE_MB" || exit 1

    # Ensure the backup directory exists
    mkdir -p "$RMSBACKUPDIR"

    # Check if a previous backup/restore cycle was interrupted
    UPDATEINPROGRESS="0"
    if [ -f "$UPDATEINPROGRESSFILE" ]; then
        echo "Reading custom files protection state..."
        UPDATEINPROGRESS=$(cat "$UPDATEINPROGRESSFILE")
        echo "Previous backup/restore cycle state: $UPDATEINPROGRESS"
    fi

    # Backup files before any modifications if no interrupted cycle
    if [ "$UPDATEINPROGRESS" = "0" ]; then
        backup_files
    else
        echo "Skipping backup due to interrupted backup/restore cycle."
    fi

    # Change to the RMS source directory
    cd "$RMSSOURCEDIR" || { echo "Error: RMS source directory not found. Exiting."; exit 1; }

    # Get current branch name
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "Current branch: $CURRENT_BRANCH"

    # Activate the virtual environment
    if [ -f ~/vRMS/bin/activate ]; then
        source ~/vRMS/bin/activate
    else
        echo "Error: Virtual environment not found. Exiting."
        exit 1
    fi

    # Perform cleanup operations before updating
    echo "Removing the build directory..."
    rm -rf build

    echo "Cleaning up Python bytecode files..."
    if command -v pyclean >/dev/null 2>&1; then
        pyclean . -v --debris all
    else
        echo "pyclean not found, using basic cleanup..."
        find . -name "*.pyc" -type f -delete
        find . -type d -name "__pycache__" -exec rm -r {} +
        find . -name "*.pyo" -type f -delete
    fi

    echo "Cleaning up *.so files in the repository..."
    find . -name "*.so" -type f -delete

    # Mark custom files backup/restore cycle as in progress
    echo "1" > "$UPDATEINPROGRESSFILE"

    # Improved Git update process with retries
    echo "Fetching updates from remote..."
    if ! git_with_retry "fetch"; then
        echo "Error: Failed to fetch updates. Aborting."
        exit 1
    fi

    # Check if updates are needed
    echo "Checking for available updates..."
    if ! git log HEAD.."origin/$CURRENT_BRANCH" --oneline | grep .; then
        echo "Local repository already up to date with origin/$CURRENT_BRANCH"
    else
        echo "Updates available, resetting to remote state..."
        if ! git_with_retry "reset" "$CURRENT_BRANCH"; then
            echo "Error: Failed to reset to origin/$CURRENT_BRANCH. Aborting."
            exit 1
        fi
        echo "Successfully updated to latest version"
    fi

    # Create template from the current default config file
    if [ -f "$CURRENT_CONFIG" ]; then
        echo "Creating config template..."
        mv "$CURRENT_CONFIG" "$RMSSOURCEDIR/.configTemplate"
        
        # Verify the move worked
        if [ ! -f "$RMSSOURCEDIR/.configTemplate" ]; then
            echo "Warning: Failed to verify config template creation"
        else
            echo "Config template created successfully"
        fi
    fi

    # Restore files after updates
    restore_files

    # Mark custom files backup/restore cycle as completed
    echo "0" > "$UPDATEINPROGRESSFILE"

    # Install missing dependencies
    install_missing_dependencies

    # Install Python requirements
    echo -e "\n========== Installing Python Requirements =========="
    pip install -r requirements.txt
    echo -e "===============================================\n"

    # Run the Python setup and suppress Cython compile noise
    python setup.py install
    

    echo "Update process completed successfully! Exiting in 5 seconds..."
    sleep 5
}

# Run the main process
main