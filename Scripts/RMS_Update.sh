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
INITIAL_COMMIT=""
INITIAL_DATE=""
FINAL_COMMIT=""
FINAL_DATE=""
CURRENT_CONFIG="$RMSSOURCEDIR/.config"
CURRENT_MASK="$RMSSOURCEDIR/mask.bmp"
CURRENT_CAMERA_SETTINGS="$RMSSOURCEDIR/camera_settings.json"
BACKUP_CONFIG="$RMSBACKUPDIR/.config"
BACKUP_MASK="$RMSBACKUPDIR/mask.bmp"
BACKUP_CAMERA_SETTINGS="$RMSBACKUPDIR/camera_settings.json"
SYSTEM_PACKAGES="$RMSSOURCEDIR/system_packages.txt"
UPDATEINPROGRESSFILE=$RMSBACKUPDIR/update_in_progress
LOCKFILE="/tmp/rms_update.$(sha1sum <<<"$RMSSOURCEDIR" | cut -c1-8).lock"
MIN_SPACE_MB=200  # Minimum required space in MB
RETRY_LIMIT=3  # Retries for critical file operations
GIT_RETRY_LIMIT=6
GIT_RETRY_DELAY=15  # Seconds between git operation retries

usage() {
    echo "Usage: $0 [--switch <branch>] [--help]"
    echo "  --switch <branch>  Interactively switch or switch to a specific branch"
    echo "  --help             Show usage info"
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --switch)
                if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                    SWITCH_MODE="direct"
                    SWITCH_BRANCH="$2"
                    shift 2
                else
                    SWITCH_MODE="interactive"
                    shift 1
                fi
                ;;
            --help|-h)
                usage
                ;;
            *)
                echo "Unknown argument: $1"
                usage
                ;;
        esac
    done
}

# Check if we're in an interactive terminal that supports tput
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && tput colors >/dev/null 2>&1; then
    USE_COLOR=true
else
    USE_COLOR=false
fi

# Functions for improved status output
print_status() {
    local type=$1
    local msg=$2
    
    if [ "$USE_COLOR" = true ]; then
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
    else
        # Fallback for non-color terminals
        case $type in
            "error")
                echo "[ERROR] $msg"
                sleep 2
                ;;
            "warning")
                echo "[WARNING] $msg"
                sleep 1
                ;;
            "success")
                echo "[SUCCESS] $msg"
                ;;
            "info")
                echo "[INFO] $msg"
                ;;
        esac
    fi
}

print_header() {
    local msg=$1
    echo ""
    if [ "$USE_COLOR" = true ]; then
        tput bold; tput setaf 6  # Bold cyan
        echo "====== $msg ======"
        tput sgr0
    else
        echo "====== $msg ======"
    fi
    echo ""
    sleep 1
}

check_git_setup() {
    print_header "Checking Git Configuration"
    
    # Check if this is a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_status "error" "Not a git repository. Please run this script from ~/source/RMS"
        exit 1
    else
        print_status "info" "Valid git repository found"
    fi
    
    # Check what remotes we have
    print_status "info" "Checking remote configuration..."
    local remotes=$(git remote)
    local rms_url="https://github.com/CroatianMeteorNetwork/RMS.git"
    RMS_REMOTE=""  # Will store the remote we'll use
    
    if [ -z "$remotes" ]; then
        print_status "warning" "No remotes configured. Adding RMS repository..."
        git remote add origin "$rms_url"
        RMS_REMOTE="origin"
    else
        # Check all remotes to find RMS repository
        for remote in $remotes; do
            url=$(git remote get-url $remote)
            print_status "info" "Found remote '$remote' pointing to: $url"
            if [[ "$url" == *"CroatianMeteorNetwork/RMS"* ]]; then
                RMS_REMOTE="$remote"
                print_status "success" "Found RMS repository at remote '$remote'"
                break
            fi
        done
        
        # If no RMS remote found, add one
        if [ -z "$RMS_REMOTE" ]; then
            print_status "warning" "No remote points to RMS repository. Adding it..."
            git remote add rms "$rms_url"
            RMS_REMOTE="rms"
        fi
    fi
    
    # Verify we can reach the RMS repository
    print_status "info" "Verifying connection to RMS remote..."
    if ! git ls-remote --exit-code "$RMS_REMOTE" >/dev/null 2>&1; then
        print_status "error" "Cannot reach RMS repository. Please check your internet connection"
        exit 1
    else
        print_status "success" "Successfully connected to RMS repository through '$RMS_REMOTE' remote"
    fi
}


# Function to handle interactive branch selection
switch_branch_interactive() {
    print_status "info" "Fetching available branches..."
    
    # Grab the *actual* current local branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    # Get list of remote branches, excluding HEAD
    branches=( $(git branch -r | grep "$RMS_REMOTE/" | grep -v HEAD | sed "s/$RMS_REMOTE\///") )    
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

# Function to check and fix git index
check_git_index() {
    if ! git status &>/dev/null; then
        if [[ $(git status 2>&1) == *"index file"* ]]; then
            print_status "warning" "Corrupted git index detected"
            print_status "info" "Attempting to fix git index..."
            rm -f .git/index
            git reset &>/dev/null
            if ! git status &>/dev/null; then
                print_status "error" "Failed to fix git index. Manual intervention required:"
                print_status "error" "1. rm .git/index"
                print_status "error" "2. git reset"
                return 1
            fi
            print_status "success" "Git index fixed"
        else
            print_status "error" "Git repository is in an invalid state"
            return 1
        fi
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

    # Backup camera_settings.json
    if [ -f "$CURRENT_CAMERA_SETTINGS" ]; then
        if ! retry_cp "$CURRENT_CAMERA_SETTINGS" "$BACKUP_CAMERA_SETTINGS"; then
            print_status "error" "Could not back up camera_settings.json file."
        fi
    else
        print_status "info" "No original camera_settings.json found. Default settings will be used."
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

    # Restore camera_settings.json
    if [ -f "$BACKUP_CAMERA_SETTINGS" ]; then
        if ! retry_cp "$BACKUP_CAMERA_SETTINGS" "$CURRENT_CAMERA_SETTINGS"; then
            print_status "error" "Failed to restore camera_settings.json."
        fi
    else
        print_status "info" "No backup camera_settings.json found - a new default settings file will be created by the installation."
    fi
}

# Function to repair corrupted repository
repair_repository() {
    print_header "Attempting Repository Repair"
    local repair_success=false
    
    # Try basic repair first
    print_status "info" "Attempting basic repository repair..."
    git fsck --full 2>/dev/null

    # Get list of corrupted objects
    local corrupted_objects=$(find .git/objects -type f -empty | sed 's/\.git\/objects\///')
    
    if [ -n "$corrupted_objects" ]; then
        print_status "warning" "Found corrupted objects, attempting removal..."
        while IFS= read -r obj; do
            rm -f ".git/objects/$obj"
        done <<< "$corrupted_objects"
        
        # Try to repair again
        if ! git fsck --full 2>/dev/null; then
            print_status "warning" "Basic repair failed, attempting full reclone..."
            
            # Backup current directory
            local timestamp=$(date +%Y%m%d_%H%M%S)
            local backup_dir="${RMSSOURCEDIR}_backup_${timestamp}"
            
            print_status "info" "Creating backup at: $backup_dir"
            if ! mv "$RMSSOURCEDIR" "$backup_dir"; then
                print_status "error" "Failed to create backup. Aborting repair."
                return 1
            fi
            
            # Reclone repository
            print_status "info" "Recloning repository..."
            if ! git clone https://github.com/CroatianMeteorNetwork/RMS.git "$RMSSOURCEDIR"; then
                print_status "error" "Failed to reclone repository. Restoring backup..."
                mv "$backup_dir" "$RMSSOURCEDIR"
                return 1
            fi
            
            # Restore config files from backup
            print_status "info" "Restoring configuration from backup..."
            cp "$backup_dir/.config" "$RMSSOURCEDIR/" 2>/dev/null
            cp "$backup_dir/mask.bmp" "$RMSSOURCEDIR/" 2>/dev/null
            cp "$backup_dir/camera_settings.json" "$RMSSOURCEDIR/" 2>/dev/null
            
            repair_success=true
        else
            repair_success=true
        fi
    else
        print_status "info" "No empty objects found, checking repository integrity..."
        if git fsck --full 2>/dev/null; then
            repair_success=true
        fi
    fi
    
    if [ "$repair_success" = true ]; then
        print_status "success" "Repository repair completed successfully"
        return 0
    else
        print_status "error" "Repository repair failed"
        return 1
    fi
}

# Function for reliable git operations
git_with_retry() {
    local cmd=$1
    local branch=$2
    local attempt=1
    local backup_dir="${RMSSOURCEDIR}_backup_$(date +%Y%m%d_%H%M%S)"

    while [ $attempt -le $GIT_RETRY_LIMIT ]; do
        local depth_arg=""
        print_status "info" "Attempting git $cmd (try $attempt of $GIT_RETRY_LIMIT)..."

        # Step 1: Clear Any Cached Git Settings to Ensure a Clean Retry
        git config --global --unset http.version
        git config --global --unset http.sslverify
        git config --global --unset http.postbuffer

        # Step 2: Apply Recommended Git Settings Before Each Attempt
        git config --global http.version HTTP/1.1
        git config --global http.sslverify false
        git config --global http.postbuffer 1048576000  # Large buffer for large repo fetches

        case $attempt in
            2)
                print_status "info" "Switching to HTTP/1.1 for this attempt"
                ;;
            3)
                print_status "info" "Using --depth=1 for a shallow fetch"
                depth_arg="--depth=1"
                ;;
            4)
                print_status "info" "Resetting Git settings and retrying with HTTP/1.1 and --depth=1"
                git config --global --unset http.version
                git config --global http.version HTTP/1.1
                depth_arg="--depth=1"
                ;;
            5)
                print_status "warning" "Final Git attempt: Recloning repository using HTTP/1.1"
                git config --global http.version HTTP/1.1

                cd ~ || exit 1
                mv "$RMSSOURCEDIR" "$backup_dir"

                if git clone --config http.version=HTTP/1.1 https://github.com/CroatianMeteorNetwork/RMS.git "$RMSSOURCEDIR"; then
                    print_status "success" "Repository successfully recloned using HTTP/1.1"
                    cd "$RMSSOURCEDIR" || exit 1

                    # Restore critical files from backup
                    for file in .config mask.bmp camera_settings.json; do
                        if [ -f "$backup_dir/$file" ]; then
                            print_status "info" "Restoring $file from backup"
                            cp "$backup_dir/$file" "$RMSSOURCEDIR/"
                        fi
                    done
                    return 0
                else
                    print_status "error" "Reclone failed. Proceeding with wget/curl..."
                fi
                ;;
            6)
                print_status "error" "All Git attempts failed. Attempting to download GitHub tarball..."

                cd ~/source || exit 1
                mv "$RMSSOURCEDIR" "$backup_dir"

                # Step 3: Restart SSH Agent & Kill Git Processes to Ensure Clean State
                print_status "info" "Restarting SSH agent to clear any stuck Git connections..."
                eval "$(ssh-agent -s)"  # Restart SSH agent

                # Graceful shutdown
                pkill -u "$USER" -f git 2>/dev/null || true
                sleep 2

                # Hard kill if anything is left
                pkill -9 -u "$USER" -f git 2>/dev/null || true
                
                # Step 4: Attempt wget/curl with Clean Environment
                print_status "info" "Downloading latest RMS repository from GitHub using wget (supports resume)..."
                if ! wget -c --tries=5 --timeout=30 -O RMS.tar.gz https://github.com/CroatianMeteorNetwork/RMS/archive/refs/heads/master.tar.gz; then
                    print_status "warning" "wget failed. Trying curl as backup..."

                    if ! curl -L --retry 5 --retry-delay 10 --connect-timeout 30 -o RMS.tar.gz https://github.com/CroatianMeteorNetwork/RMS/archive/refs/heads/master.tar.gz; then
                        print_status "error" "Both wget and curl failed. Restoring original backup..."
                        mv "$backup_dir" "$RMSSOURCEDIR"
                        return 1
                    fi
                fi

                # Verify the tarball integrity before extracting
                if ! tar -tzf RMS.tar.gz >/dev/null; then
                    print_status "error" "Downloaded tarball is corrupted. Restoring original backup..."
                    mv "$backup_dir" "$RMSSOURCEDIR"
                    return 1
                fi

                print_status "success" "Tarball downloaded successfully. Extracting..."
                mkdir -p "$RMSSOURCEDIR"
                tar -xzf RMS.tar.gz --strip-components=1 -C "$RMSSOURCEDIR"

                # Restore critical files from backup
                for file in .config mask.bmp camera_settings.json; do
                    if [ -f "$backup_dir/$file" ]; then
                        print_status "info" "Restoring $file from backup"
                        cp "$backup_dir/$file" "$RMSSOURCEDIR/"
                    fi
                done

                return 0
                ;;
        esac

        # Ensure Git Fails Properly Before Retrying
        case $cmd in
            "fetch")
                if ! git fetch --all --prune --force --verbose $depth_arg; then
                    print_status "warning" "Git fetch failed, retrying..."
                else
                    return 0
                fi
                ;;
            "checkout")
                if ! git checkout "$branch"; then
                    print_status "warning" "Git checkout failed, retrying..."
                else
                    return 0
                fi
                ;;
            "reset")
                if ! git reset --hard "$RMS_REMOTE/$branch"; then
                    print_status "warning" "Git reset failed, retrying..."
                else
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

ensure_branch_tracking() {
    local branch=$1
    
    print_status "info" "Ensuring proper tracking for branch: $branch"
    
    # Check if branch already has tracking information
    if ! git rev-parse --abbrev-ref "$branch@{upstream}" >/dev/null 2>&1; then
        print_status "info" "Setting upstream tracking for $branch..."
        if ! git branch --set-upstream-to="$RMS_REMOTE/$branch" "$branch"; then
            print_status "warning" "Failed to set upstream tracking. You may need to run:"
            print_status "warning" "git branch --set-upstream-to=$RMS_REMOTE/$branch $branch"
            return 1
        fi
    else
        print_status "info" "Branch $branch already has proper tracking"
    fi
    return 0
}

# Function to safely switch to a specified branch
switch_to_branch() {
    local target_branch="$1"
    local from_interactive="${2:-false}"  # Optional parameter to indicate if called from interactive mode

    # Skip validation if called from interactive mode (already validated)
    if [ "$from_interactive" = "false" ]; then
        if [[ ! "$target_branch" =~ ^[a-zA-Z0-9_/-]+$ ]]; then
            print_status "error" "Invalid branch name '$target_branch'. Branch names can only contain letters, numbers, underscores, forward slashes and hyphens"
            return 1
        fi
        
        print_status "info" "Validating branch: $target_branch"
        
        # Verify remote branch exists
        if ! git rev-parse --verify -q "$RMS_REMOTE/$target_branch" >/dev/null 2>&1; then
            print_status "error" "Branch '$target_branch' not found in remote '$RMS_REMOTE'"
            print_status "info" "Available branches:"
            git branch -r | grep "$RMS_REMOTE/" | grep -v HEAD | sed "s/$RMS_REMOTE\//  /"
            return 1
        fi

        # Verify it's actually a branch (not a tag or other ref)
        if ! git show-ref --verify --quiet "refs/remotes/$RMS_REMOTE/$target_branch"; then
            print_status "error" "'$target_branch' exists but is not a valid branch"
            return 1
        fi
    fi

    print_status "info" "Attempting to switch to branch: $target_branch"

    # First try to create a tracking branch if it doesn't exist locally
    if ! git rev-parse --verify -q "$target_branch" >/dev/null 2>&1; then
        print_status "info" "Creating local tracking branch..."
        if ! git branch --track "$target_branch" "$RMS_REMOTE/$target_branch"; then
            print_status "error" "Failed to create tracking branch for $target_branch"
            return 1
        fi
    else
        # Ensure existing branch has proper tracking
        ensure_branch_tracking "$target_branch"
    fi

    # Now try to switch to the branch
    if ! git_with_retry "checkout" "$target_branch"; then
        print_status "error" "Failed to switch to branch $target_branch. This could be due to:"
        print_status "error" "- Local conflicts that need resolution"
        print_status "error" "- Insufficient permissions"
        print_status "error" "- Corrupted local repository"
        return 1
    fi

    # Verify we're actually on the right branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "$target_branch" ]; then
        print_status "error" "Branch switch verification failed. Expected: $target_branch, Got: $current_branch"
        return 1
    fi

    print_status "success" "Successfully switched to branch: $target_branch"
    return 0
}


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
    # Identify missing packages from the hardcoded list
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
        # Show prominent message for sudo
        if [ "$USE_COLOR" = true ]; then
            tput bold; tput setaf 3  # Bold yellow
            echo "
==============================================
  Sudo access needed for package installation
==============================================
"
            tput sgr0  # Reset formatting
        else
            echo "
==============================================
  Sudo access needed for package installation
==============================================
"
        fi
        sleep 2
        
        sudo apt-get update
        for package in "${missing_packages[@]}"; do
            if ! sudo apt-get install -y "$package"; then
                print_status "error" "Failed to install $package. Please install it manually."
            fi
        done
    fi
}

get_commit_info() {
    local branch=$1
    local commit
    local date
    
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    date=$(git log -1 --format="%cd" --date=local 2>/dev/null || echo "unknown")
    
    echo "$commit|$date"
}

print_update_report() {
    print_header "Update Report"
    
    local final_commit
    local final_date
    IFS='|' read -r final_commit final_date <<< "$(get_commit_info)"
    
    if [ "$USE_COLOR" = true ]; then
        tput bold
    fi
    echo "Branch update summary:"
    echo "  From: $INITIAL_BRANCH (${INITIAL_COMMIT} - ${INITIAL_DATE})"
    echo "  To:   $RMS_BRANCH (${final_commit} - ${final_date})"
    if [ "$USE_COLOR" = true ]; then
        tput sgr0
    fi
}

# Function to handle error cleanup
cleanup_on_error() {
    print_status "warning" "Error occurred, attempting to restore files..."
    restore_files
    echo "0" > "$UPDATEINPROGRESSFILE"
    exit 1
}

#######################################################
######################   MAIN   #######################
#######################################################
main() {
    parse_args "$@"

    print_header "Starting RMS Update"
    
    # Use flock for robust locking
    exec 200>"$LOCKFILE"
    if ! flock -n 200; then
        print_status "error" "Another RMS update is already running. Exiting."
        exit 1
    fi
    # Lock will be automatically released when script exits
    # Note: Lock is inherited by RMS_PostUpdate.sh via exec, ensuring
    # the entire update process is protected

    # Run space check before anything else
    print_status "info" "Checking available disk space..."
    check_disk_space "$RMSSOURCEDIR" "$MIN_SPACE_MB" || exit 1

    # Get initial commit info
    INITIAL_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    INITIAL_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    INITIAL_DATE=$(git log -1 --format="%cd" --date=local 2>/dev/null || echo "unknown")

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

    # Check Git configuration
    check_git_setup

    print_header "Updating from Git"
    if ! check_git_index; then
        exit 1
    fi

    # Activate the virtual environment
    if [ -f ~/vRMS/bin/activate ]; then
        source ~/vRMS/bin/activate
    else
        print_status "error" "Virtual environment not found. Exiting."
        exit 1
    fi

    #######################################################
    ################ DANGER ZONE START ####################
    #######################################################

    # Mark custom files backup/restore cycle as in progress
    echo "1" > "$UPDATEINPROGRESSFILE"

   # Fetch updates
    if ! git_with_retry "fetch"; then
        print_status "error" "Failed to fetch updates. Aborting."
        cleanup_on_error
    fi

    # Stash any local changes first
    print_status "info" "Stashing any local changes..."
    if ! git stash; then
        print_status "error" "git stash failed - possible repository corruption."
        print_status "info" "Attempting graceful Git recovery..."
        if ! repair_repository; then
            print_status "error" "Repo repair failed - aborting update"
            cleanup_on_error
        fi
    fi

    # Handle branch setup and switching
    if [ "$1" = "--switch" ]; then
        if [ -n "$2" ]; then
            if ! switch_to_branch "$2"; then
                cleanup_on_error
            fi
            RMS_BRANCH="$2"
        else
            switch_branch_interactive
            if ! switch_to_branch "$RMS_BRANCH" "true"; then
                cleanup_on_error
            fi
        fi
    elif [ -z "$RMS_BRANCH" ]; then
        # If no branch specified (via --switch or environment), use current
        RMS_BRANCH=$(git rev-parse --abbrev-ref HEAD || echo "master")
        print_status "info" "Using current branch: $RMS_BRANCH"
    fi

    # Check if updates are needed
    print_status "info" "Checking for available updates..."
    if ! git log HEAD.."$RMS_REMOTE/$RMS_BRANCH" --oneline | grep .; then
        print_status "success" "Local repository already up to date with $RMS_REMOTE/$RMS_BRANCH"
    else
        print_status "info" "Updates available, resetting to remote state..."
        if ! git_with_retry "reset" "$RMS_BRANCH"; then
            print_status "error" "Failed to reset to $RMS_REMOTE/$RMS_BRANCH. Aborting."
            cleanup_on_error
        fi

        # Ensure tracking information is maintained after reset
        if ! ensure_branch_tracking "$RMS_BRANCH"; then
            print_status "warning" "Failed to set branch tracking after reset"
            print_status "warning" "You may need to manually set tracking with: git branch --set-upstream-to=$RMS_REMOTE/$RMS_BRANCH $RMS_BRANCH"
        fi
       
        print_status "success" "Successfully updated to latest version"
        sleep 2
    fi

    # Create template from the current default config file
    if [ -f "$CURRENT_CONFIG" ]; then
        print_status "info" "Creating config template..."
        mv "$CURRENT_CONFIG" "$RMSSOURCEDIR/.configTemplate"
        
        # Verify the copy worked
        if [ ! -f "$RMSSOURCEDIR/.configTemplate" ]; then
            print_status "warning" "Failed to verify config template creation"
        else
            print_status "success" "Config template created successfully"
        fi
    fi

    # Create template from the current default camera_settings file
    if [ -f "$CURRENT_CAMERA_SETTINGS" ]; then
        print_status "info" "Creating camera_settings template..."
        mv "$CURRENT_CAMERA_SETTINGS" "$RMSSOURCEDIR/camera_settings_template.json"
        
        # Verify the move worked
        if [ ! -f "$RMSSOURCEDIR/camera_settings_template.json" ]; then
            print_status "warning" "Failed to verify camera settings template creation"
        else
            print_status "success" "Camera settings template created successfully"
        fi
    fi

    # Restore files after updates
    restore_files

    # Mark custom files backup/restore cycle as completed
    echo "0" > "$UPDATEINPROGRESSFILE"

    #######################################################
    ################ DANGER ZONE END ######################
    #######################################################

    # Perform cleanup operations before updating
    print_header "Cleaning Build Environment"

    # Remove any stale global or egg installs of RMS before we rebuild
    print_status "info" "Clearing old RMS installs from site-packages…"
    pip uninstall -y RMS 2>/dev/null || true

    # Resolve the active site‑packages directory inside the venv
    SITE_DIR=$(python - <<'PY'
import site, sys
print(site.getsitepackages()[0])
PY
    )

    # Remove any leftover RMS installs (eggs, .dist-info, .egg-info)
    find "$SITE_DIR" -maxdepth 1 -name 'RMS-*.egg'       -prune -exec rm -rf {} +
    find "$SITE_DIR" -maxdepth 1 -name 'RMS-*.dist-info' -prune -exec rm -rf {} +
    find "$SITE_DIR" -maxdepth 1 -name 'RMS-*.egg-info'  -prune -exec rm -rf {} +
    
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

    # Install missing dependencies
    print_header "Installing Missing Dependencies"
    install_missing_dependencies

    print_header "Installing Python Requirements"
    print_status "info" "This may take a few minutes..."
    pip install -r requirements.txt
    print_status "success" "Python requirements installed"

    print_header "Running Setup"
    print_status "info" "Building RMS (this may take a while)..."
    if ! pip install -e . --no-deps --no-build-isolation; then
        print_status "error" "Build failed. See errors above."
        exit 1
    fi
    print_status "success" "Build completed successfully"
    
    # Print the update report
    print_update_report
    print_status "success" "Update process completed successfully!"
    sleep 3
}

# Run the main process
main "$@"