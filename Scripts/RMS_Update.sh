#!/bin/bash

# This script updates the RMS code from GitHub.
# Includes error handling, retries, and ensures critical files are never lost.

# Example Usage, from ~/source/RMS:
# 1. Run script normally (uses current branch detected by Git):
#    ./Scripts/RMS_Update.sh
# 2. List branches and switch interactively:
#    ./Scripts/RMS_Update.sh --switch
# 3. Directly switch to a specified branch:
#    ./Scripts/RMS_Update.sh --switch prerelease
# 4. Use an environment variable to specify the branch before running:
#    RMS_BRANCH=prerelease ./Scripts/RMS_Update.sh

# Directories, files, and variables
RMS_BRANCH="${RMS_BRANCH:-""}"  # Use environment variable if set, otherwise empty
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
RETRY_LIMIT=3  # Retries for critical file operations
GIT_RETRY_LIMIT=5
GIT_RETRY_DELAY=60  # Seconds between git operation retries


# Functions for improved status output
print_status() {
    local type=$1
    local msg=$2
    case $type in
        "error")
            tput bold; tput setaf 1  # Bold red
            echo "ERROR: $msg"
            sleep 2  # Longer pause for errors
            ;;
        "warning")
            tput setaf 3  # Yellow
            echo "WARNING: $msg"
            sleep 1
            ;;
        "success")
            tput setaf 2  # Green
            echo "$msg"
            ;;
        "info")
            tput setaf 6  # Cyan
            echo "$msg"
            ;;
    esac
    tput sgr0  # Reset formatting
}

print_header() {
    local msg=$1
    echo -e "\n"
    tput bold; tput setaf 6  # Bold cyan
    echo "====== $msg ======"
    tput sgr0
    echo -e "\n"
    sleep 1
}


# Function to handle interactive branch selection
switch_branch_interactive() {
    print_status "info" "Fetching available branches..."
    # First ensure we have latest branch info
    if ! git fetch --all; then
        print_status "error" "Failed to fetch branch information"
        exit 1
    fi
    
    # Grab the *actual* current local branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    # Get list of remote branches, excluding HEAD
    branches=( $(git branch -r | grep -v HEAD | sed 's/origin\///') )
    
    if [ ${#branches[@]} -eq 0 ]; then
        print_status "error" "No branches found"
        exit 1
    fi
    
    print_header "Available Branches"
    for i in "${!branches[@]}"; do
        local branch="${branches[$i]}"
        # Compare with the actual current local branch
        if [ "$branch" = "$current_branch" ]; then
            # Highlight the current branch
            tput bold; tput setaf 2
            echo "$((i+1)). ${branch} (current)"
            tput sgr0
        else
            echo "$((i+1)). ${branch}"
        fi
    done
    
    read -p "Enter the number of the branch to switch to (press Enter to keep current): " choice
    
    # Handle empty input (keep current branch)
    if [ -z "$choice" ]; then
        print_status "info" "Keeping current branch: $current_branch"
        RMS_BRANCH="$current_branch"
        return
    fi
    
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#branches[@]} )); then
        RMS_BRANCH="${branches[$((choice-1))]}"
        print_status "success" "Switched to branch: $RMS_BRANCH"
    else
        print_status "error" "Invalid selection. Exiting."
        exit 1
    fi
}

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

    print_status "info" "Available disk space: ${available_mb}MB (need ${required_mb}MB)"
    
    if [ "$available_mb" -lt "$required_mb" ]; then
        print_status "error" "Insufficient disk space in $dir. Need ${required_mb}MB, have ${available_mb}MB"
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
                print_status "warning" "Validation failed. Retrying..."
                rm -f "$temp_dest"
            fi
        else
            print_status "warning" "Copy failed. Retrying..."
            rm -f "$temp_dest"
        fi
        retries=$((retries + 1))
        sleep 1
    done

    print_status "error" "Failed to copy $src to $dest after $RETRY_LIMIT retries."
    return 1
}

# Backup files
backup_files() {
    print_header "Backing Up Original Files"

    # Backup .config
    if [ -f "$CURRENT_CONFIG" ]; then
        if ! retry_cp "$CURRENT_CONFIG" "$BACKUP_CONFIG"; then
            print_status "error" "Could not back up .config file. Aborting."
            exit 1
        fi
    else
        print_status "info" "No original .config found. Generic config will be used."
    fi

    # Backup mask.bmp
    if [ -f "$CURRENT_MASK" ]; then
        if ! retry_cp "$CURRENT_MASK" "$BACKUP_MASK"; then
            print_status "error" "Could not back up mask.bmp file. Aborting."
            exit 1
        fi
    else
        print_status "info" "No original mask.bmp found. Blank mask will be used."
    fi
}

# Restore files
restore_files() {
    print_header "Restoring Configuration Files"

    # Restore .config
    if [ -f "$BACKUP_CONFIG" ]; then
        if ! retry_cp "$BACKUP_CONFIG" "$CURRENT_CONFIG"; then
            print_status "error" "Failed to restore .config. Aborting."
            exit 1
        fi
    else
        print_status "info" "No backup .config found - a new one will be created by the installation."
    fi

    # Restore mask.bmp
    if [ -f "$BACKUP_MASK" ]; then
        if ! retry_cp "$BACKUP_MASK" "$CURRENT_MASK"; then
            print_status "error" "Failed to restore mask.bmp. Aborting."
            exit 1
        fi
    else
        print_status "info" "No backup mask.bmp found - a new blank mask will be created by the installation."
    fi
}

# Function for reliable git operations
git_with_retry() {
    local cmd=$1
    local branch=$2
    local attempt=1
    
    while [ $attempt -le $GIT_RETRY_LIMIT ]; do
        print_status "info" "Attempting git $cmd (try $attempt of $GIT_RETRY_LIMIT)..."
        
        case $cmd in
            "fetch")
                if git fetch --all --prune --force --verbose; then
                    return 0
                fi
                ;;
            "checkout")
                if git checkout "$branch"; then
                    return 0
                fi
                ;;
            "reset")
                if git reset --hard "origin/$branch"; then
                    return 0
                fi
                ;;
            *)
                print_status "error" "Unknown git command: $cmd"
                return 1
                ;;
        esac
        
        print_status "warning" "Git $cmd failed, waiting ${GIT_RETRY_DELAY}s before retry..."
        sleep $GIT_RETRY_DELAY
        attempt=$((attempt + 1))
    done
    
    print_status "error" "Git $cmd failed after $GIT_RETRY_LIMIT attempts"
    return 1
}

# Install missing dependencies
install_missing_dependencies() {
    print_status "info" "Checking system_packages file: $SYSTEM_PACKAGES"
    if [ ! -f "$SYSTEM_PACKAGES" ]; then
        print_status "warning" "System packages file not found: $SYSTEM_PACKAGES"
        return
    fi

    print_status "info" "Reading packages file..."
    cat "$SYSTEM_PACKAGES"  # Show content of file

    local missing_packages=()

    # Identify missing packages
    while read -r pkg; do
        # Skip blank lines or commented lines
        [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
        
        print_status "info" "Checking package: $pkg"
        if ! dpkg -s "$pkg" &>/dev/null; then
            print_status "info" "Package $pkg is missing"
            missing_packages+=("$pkg")
        else
            print_status "info" "Package $pkg is already installed"
        fi
    done < $SYSTEM_PACKAGES
    
    # If no missing packages, inform and return
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_status "success" "All required packages are already installed."
        return
    fi

    print_status "info" "The following packages will be installed: ${missing_packages[*]}"
    sleep 1

    if sudo -n true 2>/dev/null; then
        print_status "info" "Passwordless sudo available. Installing missing packages..."
        sudo apt-get update
        for package in "${missing_packages[@]}"; do
            if ! sudo apt-get install -y "$package"; then
                print_status "error" "Failed to install $package. Please install it manually."
            fi
        done
    else
        # Clear screen and show prominent message for sudo
        # tput clear  # Clear screen
        tput bold; tput setaf 3  # Bold yellow
        echo "
==============================================
  Sudo access needed for package installation
==============================================
"
        tput sgr0  # Reset formatting
        sleep 2
        
        sudo apt-get update
        for package in "${missing_packages[@]}"; do
            if ! sudo apt-get install -y "$package"; then
                print_status "error" "Failed to install $package. Please install it manually."
            fi
        done
    fi
}

main() {
    print_header "Starting RMS Update"
    
    # Check for running instance FIRST
    if [ -f "$LOCKFILE" ]; then
        LOCK_PID=$(cat "$LOCKFILE")
        if ps -p "$LOCK_PID" > /dev/null 2>&1; then
            print_status "error" "Another instance of the script is already running. Exiting."
            exit 1
        else
            print_status "warning" "Stale lock file found. Removing it and continuing."
            rm -f "$LOCKFILE"
        fi
    fi

    # Create lock file immediately
    echo $$ > "$LOCKFILE"
    trap cleanup EXIT

    # Run space check before anything else
    print_status "info" "Checking available disk space..."
    check_disk_space "$RMSSOURCEDIR" "$MIN_SPACE_MB" || exit 1

    # Ensure the backup directory exists
    mkdir -p "$RMSBACKUPDIR"

    # Check if a previous backup/restore cycle was interrupted
    UPDATEINPROGRESS="0"
    if [ -f "$UPDATEINPROGRESSFILE" ]; then
        print_status "info" "Reading custom files protection state..."
        UPDATEINPROGRESS=$(cat "$UPDATEINPROGRESSFILE")
        print_status "info" "Previous backup/restore cycle state: $UPDATEINPROGRESS"
    fi

    # Backup files before any modifications if no interrupted cycle
    if [ "$UPDATEINPROGRESS" = "0" ]; then
        backup_files
    else
        print_status "warning" "Skipping backup due to interrupted backup/restore cycle."
    fi

    # Change to the RMS source directory
    cd "$RMSSOURCEDIR" || { print_status "error" "RMS source directory not found. Exiting."; exit 1; }

     # Stash any local changes first
    print_status "info" "Stashing any local changes..."
    if ! git stash; then
        print_status "warning" "Git stash failed. Proceeding with operations."
    fi

    # Handle branch setup and switching
    if [ "$1" = "--switch" ]; then
        if [ -n "$2" ]; then
            # Verify the specified branch exists
            if git fetch origin "$2" 2>/dev/null; then
                RMS_BRANCH="$2"
                if ! git_with_retry "checkout" "$RMS_BRANCH"; then
                    print_status "error" "Failed to switch to branch $RMS_BRANCH"
                    exit 1
                fi
                print_status "success" "Switched to branch: $RMS_BRANCH"
            else
                print_status "error" "Branch '$2' not found"
                exit 1
            fi
        else
            switch_branch_interactive
            if ! git_with_retry "checkout" "$RMS_BRANCH"; then
                print_status "error" "Failed to switch to branch $RMS_BRANCH"
                exit 1
            fi
        fi
    elif [ -z "$RMS_BRANCH" ]; then
        # If no branch specified (via --switch or environment), use current
        RMS_BRANCH=$(git rev-parse --abbrev-ref HEAD || echo "master")
        print_status "info" "Using current branch: $RMS_BRANCH"
    fi

    # Verify we're on the right branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    print_status "info" "Current branch: $CURRENT_BRANCH, target branch: $RMS_BRANCH"

    # Activate the virtual environment
    if [ -f ~/vRMS/bin/activate ]; then
        source ~/vRMS/bin/activate
    else
        print_status "error" "Virtual environment not found. Exiting."
        exit 1
    fi

    # Perform cleanup operations before updating
    print_header "Cleaning Build Environment"
    
    print_status "info" "Removing build directory..."
    rm -rf build

    # Clean Python bytecode files
    print_status "info" "Cleaning up Python bytecode files..."
    if command -v pyclean >/dev/null 2>&1; then
        if pyclean --help 2>&1 | grep -q -- "--debris"; then
            if ! pyclean . -v --debris all; then
                print_status "warning" "pyclean with debris failed, falling back to basic cleanup..."
                if ! pyclean .; then
                    print_status "warning" "pyclean failed, falling back to manual cleanup..."
                    find . -name "*.pyc" -type f -delete
                    find . -type d -name "__pycache__" -exec rm -r {} +
                    find . -name "*.pyo" -type f -delete
                fi
            fi
        else
            print_status "info" "pyclean basic version detected..."
            if ! pyclean .; then
                print_status "warning" "pyclean failed, falling back to manual cleanup..."
                find . -name "*.pyc" -type f -delete
                find . -type d -name "__pycache__" -exec rm -r {} +
                find . -name "*.pyo" -type f -delete
            fi
        fi
    else
        print_status "info" "pyclean not found, using manual cleanup..."
        find . -name "*.pyc" -type f -delete
        find . -type d -name "__pycache__" -exec rm -r {} +
        find . -name "*.pyo" -type f -delete
    fi

    print_status "info" "Cleaning up *.so files..."
    find . -name "*.so" -type f -delete

    # Mark custom files backup/restore cycle as in progress
    echo "1" > "$UPDATEINPROGRESSFILE"

    print_header "Updating from Git"
    if ! git_with_retry "fetch"; then
        print_status "error" "Failed to fetch updates. Aborting."
        exit 1
    fi

    # Check if updates are needed
    print_status "info" "Checking for available updates..."
    if ! git log HEAD.."origin/$RMS_BRANCH" --oneline | grep .; then
        print_status "success" "Local repository already up to date with origin/$RMS_BRANCH"
    else
        print_status "info" "Updates available, resetting to remote state..."
        if ! git_with_retry "reset" "$RMS_BRANCH"; then
            print_status "error" "Failed to reset to origin/$RMS_BRANCH. Aborting."
            exit 1
        fi
        print_status "success" "Successfully updated to latest version"
        sleep 2
    fi

    # Create template from the current default config file
    if [ -f "$CURRENT_CONFIG" ]; then
        print_status "info" "Creating config template..."
        mv "$CURRENT_CONFIG" "$RMSSOURCEDIR/.configTemplate"
        
        # Verify the move worked
        if [ ! -f "$RMSSOURCEDIR/.configTemplate" ]; then
            print_status "warning" "Failed to verify config template creation"
        else
            print_status "success" "Config template created successfully"
        fi
    fi

    # Restore files after updates
    restore_files

    # Mark custom files backup/restore cycle as completed
    echo "0" > "$UPDATEINPROGRESSFILE"

    # Install missing dependencies
    # -----------------------------------------------------------------------------
    # We store system-level dependencies in a separate file (system_packages.txt)
    # so that when RMS_Update pulls new code (including a potentially updated list of packages),
    # we can read those new dependencies during the same run — no need to run the update
    # script twice. Because the main script is loaded into memory, changing it mid-run
    # won't reload it. But updating this separate file allows us to immediately pick
    # up any added or changed packages without requiring a second pass.
    # -----------------------------------------------------------------------------
    print_header "Installing Missing Dependencies"
    install_missing_dependencies

    print_header "Installing Python Requirements"
    print_status "info" "This may take a few minutes..."
    pip install -r requirements.txt
    print_status "success" "Python requirements installed"

    print_header "Running Setup"
    print_status "info" "Building RMS (this may take a while)..."
    python setup.py install
    print_status "success" "Build completed successfully"

    print_status "success" "Update process completed successfully!"
    sleep 3
}

# Run the main process
main "$@"