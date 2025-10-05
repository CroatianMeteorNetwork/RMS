#!/bin/bash

# This script updates the RMS code from GitHub.
# Includes error handling, retries, and ensures critical files are never lost.

# Directories, files, and variables
RMSSOURCEDIR=~/source/RMS
RMSBACKUPDIR=~/.rms_backup
CURRENT_CONFIG="$RMSSOURCEDIR/.config"
CURRENT_MASK="$RMSSOURCEDIR/mask.bmp"
CURRENT_CAMERA_SETTINGS="$RMSSOURCEDIR/camera_settings.json"
BACKUP_CONFIG="$RMSBACKUPDIR/.config"
BACKUP_MASK="$RMSBACKUPDIR/mask.bmp"
BACKUP_CAMERA_SETTINGS="$RMSBACKUPDIR/camera_settings.json"
SYSTEM_PACKAGES="$RMSSOURCEDIR/system_packages.txt"
UPDATEINPROGRESSFILE=$RMSBACKUPDIR/update_in_progress
LOCKFILE="$RMSBACKUPDIR/update.lock"
MIN_SPACE_MB=200  # Minimum required space in MB
RETRY_LIMIT=3

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

# Run space check before anything else
echo "Checking available disk space..."
check_disk_space "$RMSSOURCEDIR" "$MIN_SPACE_MB" || exit 1

# Function to clean up and release the lock on exit
cleanup() {
    rm -f "$LOCKFILE"
}

# Ensure only one instance of the script runs at a time
if [ -f "$LOCKFILE" ]; then
    # Read the PID from the lock file
    LOCK_PID=$(cat "$LOCKFILE")
    
    # Check if the process is still running
    if ps -p "$LOCK_PID" > /dev/null 2>&1; then
        echo "Another instance of the script is already running. Exiting."
        exit 1
    else
        echo "Stale lock file found. Removing it and continuing."
        rm -f "$LOCKFILE"
    fi
fi

# Create a lock file with the current process ID
echo $$ > "$LOCKFILE"
trap cleanup EXIT

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
            echo "Critical Error: Could not back up .config file."
        fi
    else
        echo "No original .config found. Generic config will be used."
    fi

    # Backup mask.bmp
    if [ -f "$CURRENT_MASK" ]; then
        if ! retry_cp "$CURRENT_MASK" "$BACKUP_MASK"; then
            echo "Critical Error: Could not back up mask.bmp file."
        fi
    else
        echo "No original mask.bmp found. Blank mask will be used."
    fi

    # Backup camera_settings.json
    if [ -f "$CURRENT_CAMERA_SETTINGS" ]; then
        if ! retry_cp "$CURRENT_CAMERA_SETTINGS" "$BACKUP_CAMERA_SETTINGS"; then
            echo "Critical Error: Could not back up camera_settings.json file."
        fi
    else
        echo "No original camera_settings.json found. Blank mask will be used."
    fi
}

# Restore files
restore_files() {
    echo "Restoring configuration and mask files..."

    # Restore .config
    if [ -f "$BACKUP_CONFIG" ]; then
        if ! retry_cp "$BACKUP_CONFIG" "$CURRENT_CONFIG"; then
            echo "Critical Error: Failed to restore .config."
        fi
    else
        echo "No backup .config found - a new one will be created by the installation."
    fi

    # Restore mask.bmp
    if [ -f "$BACKUP_MASK" ]; then
        if ! retry_cp "$BACKUP_MASK" "$CURRENT_MASK"; then
            echo "Critical Error: Failed to restore mask.bmp."
        fi
    else
        echo "No backup mask.bmp found - a new blank mask will be created by the installation."
    fi

    # Restore camera_settings.json
    if [ -f "$BACKUP_CAMERA_SETTINGS" ]; then
        if ! retry_cp "$BACKUP_CAMERA_SETTINGS" "$CURRENT_CAMERA_SETTINGS"; then
            echo "Critical Error: Failed to restore camera_settings.json."
        fi
    else
        echo "No backup camera_settings.json found - a new default settings file will be created by the installation."
    fi
}



recover_git_repo_gracefully() {
    echo "Gracefully recovering RMS Git repository..."

    backup_files

    echo "Removing corrupted .git directory..."
    rm -rf .git

    echo "Reinitializing Git repository..."
    git init
    git remote add origin https://github.com/CroatianMeteorNetwork/RMS.git
    git fetch
    git reset --hard origin/master

    restore_files

    echo "Git recovery complete."
}



# Ensure the backup directory exists
mkdir -p "$RMSBACKUPDIR"

# Check if the update was interrupted previously
UPDATEINPROGRESS="0"
if [ -f "$UPDATEINPROGRESSFILE" ]; then
    echo "Reading update in progress file..."
    UPDATEINPROGRESS=$(cat "$UPDATEINPROGRESSFILE")
    echo "Update interruption status: $UPDATEINPROGRESS"
fi

# Backup files before any modifications
if [ "$UPDATEINPROGRESS" = "0" ]; then
    backup_files
else
    echo "Skipping backup due to interrupted update state."
fi

# Change to the RMS source directory
cd "$RMSSOURCEDIR" || { echo "Error: RMS source directory not found. Exiting."; exit 1; }

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
    # Remove .pyc files
    find . -name "*.pyc" -type f -delete
    # Remove __pycache__ directories
    find . -type d -name "__pycache__" -exec rm -r {} +
    # Remove .pyo files if they exist
    find . -name "*.pyo" -type f -delete
fi

echo "Cleaning up *.so files in the repository..."
find . -name "*.so" -type f -delete

# Mark the update as in progress
echo "1" > "$UPDATEINPROGRESSFILE"

# Stash any local changes
echo "Stashing local changes..."
if ! git stash; then
    echo "Error: git stash failed - possible repository corruption."

    echo "Attempting to restore backed up files..."
    if restore_files; then
        echo "Files restored successfully."
        echo "0" > "$UPDATEINPROGRESSFILE"
    else
        echo "Critical: File restore failed. Leaving update flag set."
    fi


    echo "Attempting graceful Git recovery..."
    recover_git_repo_gracefully

fi

# Pull the latest code from GitHub
echo "Pulling latest code from GitHub..."
if ! git pull; then
    echo "Error: git pull failed. Attempting to restore backed up files..."
    if restore_files; then
        echo "Files restored successfully."
        echo "0" > "$UPDATEINPROGRESSFILE"
    else
        echo "Critical: File restore failed. Leaving update flag set."
    fi
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

# Create template from the current default camera_settings file
if [ -f "$CURRENT_CAMERA_SETTINGS" ]; then
    echo "Creating camera_settings template..."
    mv "$CURRENT_CAMERA_SETTINGS" "$RMSSOURCEDIR/camera_settings_template.json"
    
    # Verify the move worked
    if [ ! -f "$RMSSOURCEDIR/camera_settings_template.json" ]; then
        echo "Warning: Failed to verify camera settings template creation"
    else
        echo "Camera settings template created successfully"
    fi
fi

# Install missing dependencies
install_missing_dependencies() {

    if [ ! -f "$SYSTEM_PACKAGES" ]; then
        echo "Warning: System packages file not found: $SYSTEM_PACKAGES"
        return
    fi

    local missing_packages=()

    # -----------------------------------------------------------------------------
    # We store system-level dependencies in a separate file (system_packages.txt)
    # so that when RMS_Update pulls new code (including a potentially updated list of packages),
    # we can read those new dependencies during the same run - no need to run the update
    # script twice. Because the main script is loaded into memory, changing it mid-run
    # won't reload it. But updating this separate file allows us to immediately pick
    # up any added or changed packages without requiring a second pass.
    # -----------------------------------------------------------------------------

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
        echo "sudo privileges required. Prompting for password."
        sudo apt-get update
        for package in "${missing_packages[@]}"; do
            if ! sudo apt-get install -y "$package"; then
                echo "Failed to install $package. Please install it manually."
            fi
        done
    fi
}

install_missing_dependencies

# Install Python requirements
pip install -r requirements.txt

# Run the Python setup
python setup.py install

# Restore files after updates
restore_files

# Mark the update as completed
echo "0" > "$UPDATEINPROGRESSFILE"

echo "Update process completed successfully! Exiting in 5 seconds..."
sleep 5
