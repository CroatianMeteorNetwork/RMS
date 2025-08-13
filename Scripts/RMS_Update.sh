#!/usr/bin/env bash

# Enable strict error handling
set -Eeuo pipefail

# Enable ERR propagation into functions/subshells when possible (Bash 4.4+)
shopt -s inherit_errexit 2>/dev/null || true

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

# Directories, files, and variables - marked readonly for safety
RMS_BRANCH="${RMS_BRANCH:-""}"  # Use environment variable if set, otherwise empty
SWITCH_MODE=""  # Set by parse_args: "", "direct", or "interactive"
FORCE_UPDATE=false  # Set by parse_args when --force is used
readonly RMSSOURCEDIR=$HOME/source/RMS
readonly RMSBACKUPDIR=$HOME/.rms_backup
readonly CURRENT_CONFIG="$RMSSOURCEDIR/.config"
readonly CURRENT_MASK="$RMSSOURCEDIR/mask.bmp"
readonly CURRENT_CAMERA_SETTINGS="$RMSSOURCEDIR/camera_settings.json"
BACKUP_CONFIG="$RMSBACKUPDIR/.config"  # Mutable - can change to alt locations
BACKUP_MASK="$RMSBACKUPDIR/mask.bmp"    # Mutable - can change to alt locations
BACKUP_CAMERA_SETTINGS="$RMSBACKUPDIR/camera_settings.json"  # Mutable - can change to alt locations
readonly SYSTEM_PACKAGES="$RMSSOURCEDIR/system_packages.txt"
readonly BACKUP_STATE_FILE=$RMSBACKUPDIR/backup_state
readonly LOCKFILE="/tmp/rms_update.$(printf '%s' "$RMSSOURCEDIR" | sha1sum | cut -c1-8).lock"
readonly MIN_SPACE_MB=200  # Minimum required space in MB
readonly RETRY_LIMIT=3  # Retries for critical file operations
readonly GIT_RETRY_LIMIT=5
readonly GIT_RETRY_DELAY=15  # Seconds between git operation retries

# Trap handler for emergency cleanup
emergency_cleanup() {
    local exit_code=$?
    local line_number=${1:-"unknown"}
    
    # Prevent re-entrancy by disabling the trap immediately
    trap - ERR INT TERM
    
    # Disable strict error handling for cleanup operations
    set +e +u
    set +o pipefail
    
    # Release lock safely before any potentially failing operations
    exec 200>&- 2>/dev/null || true
    
    print_status "error" "Script failed at line $line_number with exit code $exit_code"
    print_status "warning" "Attempting emergency recovery..."
    
    # Only try to restore from backup if we were in the danger zone
    if [ -f "$BACKUP_STATE_FILE" ]; then
        local backup_state
        backup_state=$(cat "$BACKUP_STATE_FILE" 2>/dev/null || echo "unknown")
        if [ "$backup_state" = "1" ]; then
            print_status "info" "Backup state indicates we were in danger zone, attempting restoration..."
            restore_files 2>/dev/null || print_status "warning" "Failed to restore backup files"
        else
            print_status "info" "No restoration needed - backup state: $backup_state"
        fi
    fi
    
    # Try to reset git state if we're in a bad state - but don't fail if it doesn't work
    if [ -d "$RMSSOURCEDIR/.git" ]; then
        cd "$RMSSOURCEDIR" 2>/dev/null || true
        if git status &>/dev/null 2>&1; then
            print_status "info" "Attempting to restore git state..."
            check_and_fix_git_state 2>/dev/null || print_status "warning" "Failed to fix git state"
        fi
    fi
    
    print_status "error" "Update failed. Repository may be in an inconsistent state."
    print_status "info" "Manual recovery may be required. Check ~/source/RMS/.git/logs/ for git history."
    
    exit $exit_code
}

# Set up trap for errors and signals
trap 'emergency_cleanup $LINENO' ERR INT TERM

# Safety check to prevent catastrophic deletions if RMSSOURCEDIR is misconfigured
validate_rms_directory() {
    local dir="$1"
    
    # Check if directory is obviously dangerous
    case "$dir" in
        "/" | "/bin" | "/usr" | "/etc" | "/var" | "/home" | "/root" | "/sys" | "/proc" | "/dev")
            print_status "error" "RMSSOURCEDIR points to dangerous system directory: $dir"
            return 1
            ;;
    esac
    
    # Check if it contains RMS-like files (basic sanity check)
    if [ -d "$dir" ] && [ ! -f "$dir/RMS/ConfigReader.py" ] && [ ! -f "$dir/setup.py" ]; then
        print_status "warning" "Directory $dir doesn't appear to contain RMS code"
        print_status "warning" "Expected to find RMS/ConfigReader.py or setup.py"
        return 1
    fi
    
    return 0
}

usage() {
    print_status "info" "Usage: $0 [--switch <branch>] [--force] [--help]"
    print_status "info" "  --switch <branch>  Interactively switch or switch to a specific branch"
    print_status "info" "  --force            Force update even if repository is up-to-date"
    print_status "info" "  --help             Show usage info"
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --switch)
                if [[ $# -gt 1 && -n "$2" && ! "$2" =~ ^-- ]]; then
                    SWITCH_MODE="direct"
                    # Trim whitespace from branch name
                    SWITCH_BRANCH=$(echo "$2" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                    shift 2
                else
                    SWITCH_MODE="interactive"
                    shift 1
                fi
                ;;
            --force)
                FORCE_UPDATE=true
                shift 1
                ;;
            --help|-h)
                usage
                ;;
            *)
                print_status "error" "Unknown argument: $1"
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

# Check if we're running interactively (for sleep timing)
if [ -t 0 ] && [ -t 1 ]; then
    INTERACTIVE=true
else
    INTERACTIVE=false
fi

# Conditional sleep - only sleep when interactive
interactive_sleep() {
    if [ "$INTERACTIVE" = true ]; then
        sleep "$1"
    fi
}

# Functions for improved status output
print_status() {
    local type=$1
    local msg=$2
    
    if [ "$USE_COLOR" = true ]; then
        case $type in
            "error")
                tput bold; tput setaf 1  # Bold red
                echo "ERROR: $msg"
                interactive_sleep 2  # Longer pause for errors
                ;;
            "warning")
                tput setaf 3  # Yellow
                echo "WARNING: $msg"
                interactive_sleep 1
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
                interactive_sleep 2
                ;;
            "warning")
                echo "[WARNING] $msg"
                interactive_sleep 1
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
    interactive_sleep 1
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
    # Check if we're in an interactive terminal
    if [ "$INTERACTIVE" = false ]; then
        print_status "error" "Interactive switch needs a TTY. Use '--switch <branch>' or set RMS_BRANCH."
        exit 1
    fi
    
    print_status "info" "Fetching available branches..."
    
    # Fetch fresh branch list from remote
    if ! git_with_retry "fetch"; then
        print_status "error" "Failed to fetch branches"
        exit 1
    fi
    
    # Grab the *actual* current local branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    # Get list of remote branches, excluding symbolic references
    local -a branches
    readarray -t branches < <(
        git branch -r \
        | sed 's/^[[:space:]]*//' \
        | grep "^${RMS_REMOTE}/" \
        | grep -v ' -> ' \
        | sed "s|^${RMS_REMOTE}/||"
    )    
    if [ ${#branches[@]} -eq 0 ]; then
        print_status "error" "No branches found"
        exit 1
    fi
    
    local attempts=0
    local max_attempts=3
    
    while (( attempts < max_attempts )); do
        print_header "Available Branches"
        for i in "${!branches[@]}"; do
            local branch="${branches[$i]}"
            # Compare with the actual current local branch
            if [ "$branch" = "$current_branch" ]; then
                # Highlight the current branch
                if [ "$USE_COLOR" = true ]; then
                    tput bold; tput setaf 2
                    echo "$((i+1)). ${branch} (current)"
                    tput sgr0
                else
                    echo "$((i+1)). ${branch} (current)"
                fi
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
            print_status "success" "Selected branch: $RMS_BRANCH"
            if git rev-parse --abbrev-ref "${RMS_REMOTE}/${RMS_BRANCH}" >/dev/null 2>&1; then
                print_status "info" "Will track upstream: ${RMS_REMOTE}/${RMS_BRANCH}"
            fi
            return
        else
            ((attempts++))
            if (( attempts < max_attempts )); then
                print_status "error" "Invalid selection. Please try again (attempt $attempts/$max_attempts)."
                interactive_sleep 1
            else
                print_status "error" "Invalid selection after $max_attempts attempts. Using current branch."
                RMS_BRANCH="$current_branch"
                return
            fi
        fi
    done
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

check_and_fix_git_state() {
    print_status "info" "Checking git repository state..."
    
    # Try to save any work first, but don't fail if stash fails
    print_status "info" "Attempting to stash any local changes..."

    # Check if there are actually changes to stash (only tracked files)
    if ! git diff --quiet || ! git diff --cached --quiet; then
        if git -c user.name=RMS-Update -c user.email=rms-update@local stash push -m "RMS auto-stash before update" 2>/dev/null; then
            print_status "warning" "Local changes to tracked files were stashed. Run 'git stash pop' after the update to recover them."
        else
            print_status "warning" "Failed to stash changes - some modifications may be lost during update"
        fi
    else
        print_status "info" "No tracked file changes to stash"
    fi
    
    # Warn about untracked files
    if git status --porcelain | grep -q '^??'; then
        print_status "info" "Untracked files detected - these will be preserved during update"
    fi
    
    # Check for merge conflicts - try gentle abort first
    if git status --porcelain | grep -q "^UU\|^AA\|^DD"; then
        print_status "warning" "Merge conflicts detected - attempting to abort merge"
        if git merge --abort 2>/dev/null; then
            print_status "info" "Merge aborted successfully"
        else
            print_status "warning" "Merge abort failed - forcing reset to HEAD"
            git reset --hard HEAD 2>/dev/null || true
        fi
    fi
    
    # Check for ongoing rebase - try gentle abort first
    if [ -d ".git/rebase-merge" ] || [ -d ".git/rebase-apply" ]; then
        print_status "warning" "Rebase in progress - attempting to abort rebase"
        if git rebase --abort 2>/dev/null; then
            print_status "info" "Rebase aborted successfully"
        else
            print_status "warning" "Rebase abort failed - forcing reset to HEAD"
            git reset --hard HEAD 2>/dev/null || true
        fi
    fi
    
    # Check for detached HEAD
    if git symbolic-ref HEAD &>/dev/null; then
        current_branch=$(git symbolic-ref --short HEAD)
        print_status "info" "On branch: $current_branch"
    else
        print_status "warning" "Repository is in detached HEAD state"
        print_status "info" "Attempting to return to a proper branch..."
        
        # Try to determine what branch we should be on
        if [ -n "$RMS_BRANCH" ]; then
            target_branch="$RMS_BRANCH"
        else
            # Default to master/main
            if git show-ref --verify --quiet refs/remotes/$RMS_REMOTE/master; then
                target_branch="master"
            elif git show-ref --verify --quiet refs/remotes/$RMS_REMOTE/main; then
                target_branch="main"
            else
                target_branch="master"  # fallback
            fi
        fi
        
        print_status "info" "Switching to branch: $target_branch"
        if ! git checkout -B "$target_branch" "$RMS_REMOTE/$target_branch" 2>/dev/null; then
            print_status "warning" "Failed to create branch, trying direct checkout..."
            git checkout "$target_branch" 2>/dev/null || true
        fi
    fi
    
    print_status "success" "Git repository state verified and cleaned"
}

# Retry mechanism for critical file operations
retry_cp() {
    local src=$1
    local dest=$2
    local temp_dest="${dest}.tmp"
    local retries=0

    while [ $retries -lt $RETRY_LIMIT ]; do
        if cp -a "$src" "$temp_dest"; then
            # Validate the copied file
            if cmp -s "$src" "$temp_dest"; then
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
    print_status "info" "Backup paths → $BACKUP_CONFIG / $BACKUP_MASK / $BACKUP_CAMERA_SETTINGS"

    # Backup .config with fallback locations
    if [ -f "$CURRENT_CONFIG" ]; then
        if ! retry_cp "$CURRENT_CONFIG" "$BACKUP_CONFIG"; then
            # Try alternative backup locations
            local alt_backup_dirs=("/tmp/rms_backup_$$" "$HOME/rms_backup_$$")
            local backup_success=false
            
            for alt_dir in "${alt_backup_dirs[@]}"; do
                mkdir -p "$alt_dir" 2>/dev/null || continue
                if retry_cp "$CURRENT_CONFIG" "$alt_dir/.config"; then
                    print_status "warning" "Backed up .config to alternative location: $alt_dir"
                    BACKUP_CONFIG="$alt_dir/.config"
                    backup_success=true
                    break
                fi
            done
            
            if [ "$backup_success" = false ]; then
                print_status "error" "Could not back up .config file to any location. Continuing without backup."
                print_status "warning" "Risk: Configuration may be lost if update fails."
            fi
        fi
    else
        print_status "info" "No original .config found. Generic config will be used."
    fi

    # Backup mask.bmp with fallback locations
    if [ -f "$CURRENT_MASK" ]; then
        if ! retry_cp "$CURRENT_MASK" "$BACKUP_MASK"; then
            # Try alternative backup locations
            local alt_backup_dirs=("/tmp/rms_backup_$$" "$HOME/rms_backup_$$")
            local backup_success=false
            
            for alt_dir in "${alt_backup_dirs[@]}"; do
                mkdir -p "$alt_dir" 2>/dev/null || continue
                if retry_cp "$CURRENT_MASK" "$alt_dir/mask.bmp"; then
                    print_status "warning" "Backed up mask.bmp to alternative location: $alt_dir"
                    BACKUP_MASK="$alt_dir/mask.bmp"
                    backup_success=true
                    break
                fi
            done
            
            if [ "$backup_success" = false ]; then
                print_status "error" "Could not back up mask.bmp file to any location. Continuing without backup."
                print_status "warning" "Risk: Mask configuration may be lost if update fails."
            fi
        fi
    else
        print_status "info" "No original mask.bmp found. Blank mask will be used."
    fi

    # Backup camera_settings.json with fallback locations
    if [ -f "$CURRENT_CAMERA_SETTINGS" ]; then
        if ! retry_cp "$CURRENT_CAMERA_SETTINGS" "$BACKUP_CAMERA_SETTINGS"; then
            # Try alternative backup locations
            local alt_backup_dirs=("/tmp/rms_backup_$$" "$HOME/rms_backup_$$")
            local backup_success=false
            
            for alt_dir in "${alt_backup_dirs[@]}"; do
                mkdir -p "$alt_dir" 2>/dev/null || continue
                if retry_cp "$CURRENT_CAMERA_SETTINGS" "$alt_dir/camera_settings.json"; then
                    print_status "warning" "Backed up camera_settings.json to alternative location: $alt_dir"
                    BACKUP_CAMERA_SETTINGS="$alt_dir/camera_settings.json"
                    backup_success=true
                    break
                fi
            done
            
            if [ "$backup_success" = false ]; then
                print_status "error" "Could not back up camera_settings.json file to any location. Continuing without backup."
                print_status "warning" "Risk: Camera settings may be lost if update fails."
            fi
        fi
    else
        print_status "info" "No original camera_settings.json found. Default settings will be used."
    fi
}

# Restore files
restore_files() {
    print_header "Restoring Configuration Files"
    print_status "info" "Restore paths ← $BACKUP_CONFIG / $BACKUP_MASK / $BACKUP_CAMERA_SETTINGS"

    # Restore .config with graceful degradation
    if [ -f "$BACKUP_CONFIG" ]; then
        if ! retry_cp "$BACKUP_CONFIG" "$CURRENT_CONFIG"; then
            print_status "error" "Failed to restore .config after $RETRY_LIMIT attempts."
            print_status "warning" "Continuing without restored config - RMS will create a new default config."
            print_status "info" "You can manually restore from: $BACKUP_CONFIG"
        fi
    else
        print_status "info" "No backup .config found - a new one will be created by the installation."
    fi

    # Restore mask.bmp with graceful degradation
    if [ -f "$BACKUP_MASK" ]; then
        if ! retry_cp "$BACKUP_MASK" "$CURRENT_MASK"; then
            print_status "error" "Failed to restore mask.bmp after $RETRY_LIMIT attempts."
            print_status "warning" "Continuing without restored mask - RMS will create a new blank mask."
            print_status "info" "You can manually restore from: $BACKUP_MASK"
        fi
    else
        print_status "info" "No backup mask.bmp found - a new blank mask will be created by the installation."
    fi

    # Restore camera_settings.json with graceful degradation
    if [ -f "$BACKUP_CAMERA_SETTINGS" ]; then
        if ! retry_cp "$BACKUP_CAMERA_SETTINGS" "$CURRENT_CAMERA_SETTINGS"; then
            print_status "error" "Failed to restore camera_settings.json after $RETRY_LIMIT attempts."
            print_status "warning" "Continuing without restored camera settings - RMS will create new default settings."
            print_status "info" "You can manually restore from: $BACKUP_CAMERA_SETTINGS"
        fi
    else
        print_status "info" "No backup camera_settings.json found - a new default settings file will be created by the installation."
    fi
}


# Function for reliable git operations
git_with_retry() {
    local cmd=$1
    local branch=${2:-""}
    local attempt=1
    local backup_dir="${RMSSOURCEDIR}_backup_$(date +%Y%m%d_%H%M%S)"

    while [ $attempt -le $GIT_RETRY_LIMIT ]; do
        # Reset variables for clean state each iteration
        local depth_arg=""
        print_status "info" "Attempting git $cmd (try $attempt of $GIT_RETRY_LIMIT)..."

        # Use ad-hoc git settings that don't persist after script exits

        # Build git config arguments for this attempt (no persistent config changes)
        local git_config_args=()
        
        case $attempt in
            2)
                print_status "info" "Switching to HTTP/1.1 for this attempt"
                git_config_args+=(-c http.version=HTTP/1.1)
                ;;
            3)
                print_status "info" "Using --depth=1 for a shallow fetch"
                depth_arg="--depth=1"
                ;;
            4)
                print_status "info" "Retrying with HTTP/1.1 and --depth=1"
                git_config_args+=(-c http.version=HTTP/1.1)
                depth_arg="--depth=1"
                ;;
            5)
                print_status "warning" "Final Git attempt: Recloning repository using HTTP/1.1"

                cd ~ || exit 1
                mv "$RMSSOURCEDIR" "$backup_dir"

                if git clone --config http.version=HTTP/1.1 --config http.sslverify=false https://github.com/CroatianMeteorNetwork/RMS.git "$RMSSOURCEDIR"; then
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
                    print_status "error" "Reclone failed after all Git recovery attempts."
                    return 1
                fi
                ;;
        esac

        # Ensure Git Fails Properly Before Retrying
        case $cmd in
            "fetch")
                if ! git "${git_config_args[@]}" fetch --all --prune --force --quiet $depth_arg; then
                    print_status "warning" "Git fetch failed, retrying..."
                else
                    return 0
                fi
                ;;
            "checkout")
                if ! git "${git_config_args[@]}" checkout "$branch"; then
                    print_status "warning" "Git checkout failed, retrying..."
                else
                    return 0
                fi
                ;;
            "reset")
                # Try reset - should work since we're only dealing with tracked files
                if git "${git_config_args[@]}" reset --hard "$RMS_REMOTE/$branch" 2>/dev/null; then
                    return 0
                else
                    print_status "warning" "Git reset failed, retrying..."
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
        if ! git rev-parse --verify --quiet "refs/remotes/$RMS_REMOTE/$target_branch" >/dev/null; then
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
    if ! git rev-parse --verify --quiet "refs/heads/$target_branch" >/dev/null; then
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
    done < "$SYSTEM_PACKAGES"

    # If no missing packages, inform and return
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_status "success" "All required packages are already installed."
        return
    fi

    print_status "info" "The following packages will be installed: ${missing_packages[*]}"
    interactive_sleep 1

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
        interactive_sleep 2
        
        sudo apt-get update
        for package in "${missing_packages[@]}"; do
            if ! sudo apt-get install -y "$package"; then
                print_status "error" "Failed to install $package. Please install it manually."
            fi
        done
    fi
}

get_commit_info() {
    local branch=${1:-""}
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
    echo "  From: $ORIGINAL_BRANCH (${ORIGINAL_COMMIT} - ${ORIGINAL_DATE})"
    echo "  To:   $RMS_BRANCH (${final_commit} - ${final_date})"
    if [ "$USE_COLOR" = true ]; then
        tput sgr0
    fi
}

# Function to handle error cleanup
cleanup_on_error() {
    print_status "warning" "Error occurred, attempting to restore files..."
    restore_files
    echo "0" > "$BACKUP_STATE_FILE"
    exit 1
}

# Function to create RMS_Reset.sh recovery script
create_recovery_script() {
    local recovery_script="$RMSSOURCEDIR/Scripts/RMS_Reset.sh"
    local current_script="$RMSSOURCEDIR/Scripts/RMS_Update.sh"
    
    print_status "info" "Creating recovery script RMS_Reset.sh..."
    
    # Copy the current script to RMS_Reset.sh
    if cp "$current_script" "$recovery_script" 2>/dev/null; then
        # Make it executable
        chmod +x "$recovery_script" 2>/dev/null || true
        print_status "success" "Recovery script created: Scripts/RMS_Reset.sh"
        print_status "info" "Use './Scripts/RMS_Reset.sh --switch' to return from old branches"
    else
        print_status "warning" "Could not create recovery script (non-critical)"
    fi
}

#######################################################
######################   MAIN   #######################
#######################################################
main() {
    parse_args "$@"

    # Protect against infinite re-exec loops (check early to know if this is a re-exec)
    REEXEC_COUNT=${REEXEC_COUNT:-0}
    export REEXEC_COUNT

    # Validate directory before attempting to enter it
    if ! validate_rms_directory "$RMSSOURCEDIR"; then
        print_status "error" "Safety check failed. Refusing to operate on directory: $RMSSOURCEDIR"
        exit 1
    fi

    # Change to RMS directory for git operations
    cd "$RMSSOURCEDIR" || { print_status "error" "RMS source directory not found. Exiting."; exit 1; }

    # Create recovery script immediately (before ANY git operations that might change RMS_Update.sh)
    create_recovery_script

    # Check Git configuration first
    check_git_setup

    # Normalize env-driven target into a direct switch unless --switch already set
    if [[ -n "$RMS_BRANCH" && -z "$SWITCH_MODE" ]]; then
        SWITCH_MODE="direct"
        SWITCH_BRANCH="${RMS_BRANCH#"$RMS_REMOTE"/}"   # strip optional remote prefix
    fi
    
    # Normalize switch branch name (strip remote prefix if present)
    [[ -n "${SWITCH_BRANCH:-}" ]] && SWITCH_BRANCH="${SWITCH_BRANCH#${RMS_REMOTE}/}"

    # ----------------- EARLY SHA CHECK -----------------
    if (( REEXEC_COUNT == 0 )) && [[ -z "$SWITCH_MODE" && "$FORCE_UPDATE" = false ]]; then
        if ! git symbolic-ref -q HEAD >/dev/null; then
            print_status "info" "Detached HEAD state detected; skipping early SHA check."
        else
            [[ -z "$RMS_BRANCH" ]] && RMS_BRANCH=$(git rev-parse --abbrev-ref HEAD)
            RMS_BRANCH=git for-each-ref --format='%(upstream:remotename)' $(git symbolic-ref -q HEAD)
            REMOTE_SHA=$(git ls-remote --quiet --heads \
                         "$RMS_REMOTE" "refs/heads/$RMS_BRANCH" | cut -f1)
            LOCAL_SHA=$(git rev-parse HEAD)

            if [[ -z "$REMOTE_SHA" ]]; then
                print_status "error" "Cannot query remote ref $RMS_BRANCH"; exit 1
            fi
            
            # Check for modified tracked files (excluding allowed config/template/mask files)
            MODIFIED_FILES=$(git diff --name-only | grep -v -E '^(\.config|camera_settings\.json|\.configTemplate|camera_settings_template\.json|mask\.bmp)$' || true)
            
            if [[ "$REMOTE_SHA" == "$LOCAL_SHA" && -z "$MODIFIED_FILES" ]]; then
                print_status "success" "Nothing new on $RMS_BRANCH and no tracked file modifications - exiting."
                exit 0
            elif [[ "$REMOTE_SHA" == "$LOCAL_SHA" && -n "$MODIFIED_FILES" ]]; then
                print_status "info" "Repository up to date but tracked files modified:"
                echo "$MODIFIED_FILES" | sed 's/^/  /'
                print_status "info" "Proceeding with update to restore tracked files"
            else
                print_status "info" "Will update ($LOCAL_SHA → $REMOTE_SHA)"
            fi
        fi
    fi
    # ---------------------------------------------------

    # Ensure RMS_BRANCH is set (in case early check was skipped)
    [[ -z "$RMS_BRANCH" ]] && RMS_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    # From here on: we know we need to do work
    print_header "Starting RMS Update"
    if (( REEXEC_COUNT >= 2 )); then
        print_status "error" "Script hash keeps changing - aborting to avoid infinite loop"
        exit 1
    fi
    
    # Capture the original script hash for self-update detection
    SELF_ABS=$(readlink -f "$0" 2>/dev/null || echo "$0")
    SCRIPT_HASH=$(sha1sum "$SELF_ABS" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
    
    # Clean up old lock files (older than 24 hours)
    find /tmp -name 'rms_update.*.lock' -mmin +1440 -delete 2>/dev/null || true
    
    # Use flock for robust locking
    exec 200>"$LOCKFILE"
    if ! flock -n 200; then
        print_status "error" "Another RMS update is already running."
        
        # Check if lock is held by a dead process (rare but possible after power loss)
        if command -v lsof >/dev/null 2>&1; then
            local lock_pids
            lock_pids=$(lsof -Fp -- "$LOCKFILE" 2>/dev/null | grep '^p' | cut -c2- || echo "")
            if [ -n "$lock_pids" ]; then
                print_status "info" "Lock held by PID(s): $lock_pids"
            else
                print_status "warning" "Lock file exists but no process found holding it."
                print_status "warning" "This may indicate a stale lock from system crash/power loss."
                print_status "info" "If no RMS update is actually running, remove: $LOCKFILE"
            fi
        fi
        exit 1
    fi
    # Let the file live; rely on flock + descriptor for locking
    # Lock will be automatically released when script exits
    # Note: Lock is inherited by re-exec via exec, ensuring
    # the entire update process is protected

    # Run space check before anything else
    print_status "info" "Checking available disk space..."
    check_disk_space "$RMSSOURCEDIR" "$MIN_SPACE_MB" || exit 1

    # Get initial commit info (preserve across re-exec)
    if [ -z "${ORIGINAL_BRANCH+x}" ]; then
        ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        ORIGINAL_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        ORIGINAL_DATE=$(git log -1 --format="%cd" --date=local 2>/dev/null || echo "unknown")
        export ORIGINAL_BRANCH ORIGINAL_COMMIT ORIGINAL_DATE
    fi

    # Ensure the backup directory exists
    mkdir -p "$RMSBACKUPDIR"
    
    # Migrate old update_in_progress file if it exists
    # TODO: This migration code can be removed after a transition period (e.g., after 2026-01-01)
    OLD_STATE_FILE="$RMSBACKUPDIR/update_in_progress"
    if [ -f "$OLD_STATE_FILE" ]; then
        print_status "info" "Migrating old backup state file..."
        mv "$OLD_STATE_FILE" "$BACKUP_STATE_FILE"
    fi

    # Check if a previous backup/restore cycle was interrupted
    BACKUP_IN_PROGRESS="0"
    if [ -f "$BACKUP_STATE_FILE" ]; then
        print_status "info" "Reading custom files protection state..."
        BACKUP_IN_PROGRESS=$(cat "$BACKUP_STATE_FILE")
        print_status "info" "Previous backup/restore cycle state: $BACKUP_IN_PROGRESS"
    fi

    # Now that we know we need to modify the work-tree, backup files if no interrupted cycle
    if [ "$BACKUP_IN_PROGRESS" = "0" ]; then
        backup_files
    else
        print_status "warning" "Skipping backup due to interrupted backup/restore cycle."
    fi

    print_header "Updating from Git"

    # Activate the virtual environment BEFORE entering the danger zone
    if [ -f ~/vRMS/bin/activate ]; then
        source ~/vRMS/bin/activate
    else
        print_status "error" "Virtual environment not found. Exiting."
        exit 1
    fi

    # Handle interactive branch selection if needed (safe - just reads remotes)
    if [ "$SWITCH_MODE" = "interactive" ]; then
        switch_branch_interactive
        print_status "info" "Target branch: $RMS_BRANCH (interactive selection)"
    fi

    #######################################################
    ################ DANGER ZONE START ####################
    #######################################################

    # Mark custom files backup/restore cycle as in progress
    echo "1" > "$BACKUP_STATE_FILE"

    if ! check_git_index; then
        cleanup_on_error
    fi

    # Check and fix problematic git states
    check_and_fix_git_state

   # Fetch updates
    if ! git_with_retry "fetch"; then
        print_status "error" "Failed to fetch updates. Aborting."
        cleanup_on_error
    fi


    # Handle branch switching (branch already determined in pre-flight)
    if [ "$SWITCH_MODE" = "direct" ]; then
        if ! switch_to_branch "$SWITCH_BRANCH"; then
            cleanup_on_error
        fi
        RMS_BRANCH=$(git rev-parse --abbrev-ref HEAD)   # Update to actual branch we're on
    elif [ "$SWITCH_MODE" = "interactive" ]; then
        if ! switch_to_branch "$RMS_BRANCH" "true"; then
            cleanup_on_error
        fi
        RMS_BRANCH=$(git rev-parse --abbrev-ref HEAD)   # Update to actual branch we're on
    fi

    # Check if updates are needed (compare HEAD to remote ref)
    print_status "info" "Checking for available updates..."
    LOCAL_SHA=$(git rev-parse HEAD)
    REMOTE_SHA=$(git rev-parse "$RMS_REMOTE/$RMS_BRANCH")
    
    # Check for modified tracked files (same logic as early check, but also exclude mask.bmp)
    MODIFIED_FILES=$(git diff --name-only | grep -v -E '^(\.config|camera_settings\.json|\.configTemplate|camera_settings_template\.json|mask\.bmp)$' || true)
    
    if [[ "$LOCAL_SHA" == "$REMOTE_SHA" && -z "$MODIFIED_FILES" ]]; then
        print_status "success" "Local repository already up to date with $RMS_REMOTE/$RMS_BRANCH"
    else
        if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
            print_status "info" "Updates available, resetting to remote state..."
        else
            print_status "info" "Repository up to date but resetting to restore modified tracked files..."
        fi
        
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
        interactive_sleep 2
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
    
    # Verify restored files uniformly: if a backup existed, the restored file must exist and match bit-for-bit
    for name in ".config" "mask.bmp" "camera_settings.json"; do
        backup="$RMSBACKUPDIR/$name"
        current="$RMSSOURCEDIR/$name"
        if [ -f "$backup" ]; then
            [[ -f "$current" ]] || { print_status "error" "$current missing after restore"; exit 1; }
            if ! cmp -s "$backup" "$current"; then
                print_status "error" "Restored $name does not match backup — aborting."
                exit 1
            fi
        fi
    done
    
    print_status "success" "All configuration files verified after restore"

    # Mark custom files backup/restore cycle as completed
    echo "0" > "$BACKUP_STATE_FILE"

    # Check if this script was updated and re-exec if needed
    SELF="$RMSSOURCEDIR/Scripts/RMS_Update.sh"
    SELF_REAL=$(readlink -f "$SELF" 2>/dev/null || echo "$SELF")
    new_hash=$(sha1sum "$SELF_REAL" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
    if [[ "$SCRIPT_HASH" != "$new_hash" ]] && [[ "$new_hash" != "unknown" ]]; then
        echo ""
        print_header "SCRIPT SELF-UPDATE DETECTED"
        print_status "warning" "RMS_Update.sh was modified during the update"
        print_status "info" "Restarting with the new version to apply improvements..."
        echo ""
        interactive_sleep 2
        # Re-exec preserves file descriptors (including our flock) and arguments
        # Handle arithmetic safely in case of unexpected values
        if next_count=$((REEXEC_COUNT+1)) 2>/dev/null; then
            exec env REEXEC_COUNT="$next_count" "$SELF" "$@"
        else
            print_status "warning" "Failed to increment REEXEC_COUNT, continuing with current version..."
        fi
    fi

    #######################################################
    ################ DANGER ZONE END ######################
    #######################################################

    # Perform cleanup operations before updating
    print_header "Cleaning Build Environment"

    # Remove any stale global or egg installs of RMS before we rebuild
    print_status "info" "Clearing old RMS installs from site-packages…"
    python -m pip uninstall -y RMS 2>/dev/null || true

    # Resolve the active site‑packages directory inside the venv
    SITE_DIR=$(python - <<'PY'
import sysconfig, sys, os
print(sysconfig.get_paths()['purelib'])
PY
    )

    # Remove any leftover RMS installs (eggs, .dist-info, .egg-info)
    find "$SITE_DIR" -maxdepth 1 -name 'RMS-*.egg'       -prune -exec rm -rf {} +
    find "$SITE_DIR" -maxdepth 1 -name 'RMS-*.dist-info' -prune -exec rm -rf {} +
    find "$SITE_DIR" -maxdepth 1 -name 'RMS-*.egg-info'  -prune -exec rm -rf {} +
    
    # Remove orphaned RMS modules from old installations
    print_status "info" "Checking for orphaned RMS modules in site-packages..."
    
    # Check if there's an RMS directory in site-packages (from old non-editable installs)
    if [ -d "$SITE_DIR/RMS" ]; then
        print_status "info" "Found old RMS directory in site-packages, removing..."
        rm -rf "$SITE_DIR/RMS"
        print_status "success" "Removed old RMS directory from site-packages"
    fi
        
    print_status "info" "Removing build and dist directories..."
    rm -rf build dist

    # Clean Python bytecode files
    print_status "info" "Cleaning up Python bytecode files..."
    if command -v pyclean >/dev/null 2>&1; then
        if pyclean --help 2>&1 | grep -q -- "--debris"; then
            if ! pyclean . -v --debris all; then
                print_status "warning" "pyclean with debris failed, falling back to basic cleanup..."
                if ! pyclean .; then
                    print_status "warning" "pyclean failed, falling back to manual cleanup..."
                    find "$RMSSOURCEDIR" -name "*.pyc" -type f -delete 2>/dev/null || true
                    find "$RMSSOURCEDIR" -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
                    find "$RMSSOURCEDIR" -name "*.pyo" -type f -delete 2>/dev/null || true
                fi
            fi
        else
            print_status "info" "pyclean basic version detected..."
            if ! pyclean .; then
                print_status "warning" "pyclean failed, falling back to manual cleanup..."
                find "$RMSSOURCEDIR" -name "*.pyc" -type f -delete 2>/dev/null || true
                find "$RMSSOURCEDIR" -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
                find "$RMSSOURCEDIR" -name "*.pyo" -type f -delete 2>/dev/null || true
            fi
        fi
    else
        print_status "info" "pyclean not found, using manual cleanup..."
        find "$RMSSOURCEDIR" -name "*.pyc" -type f -delete 2>/dev/null || true
        find "$RMSSOURCEDIR" -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
        find "$RMSSOURCEDIR" -name "*.pyo" -type f -delete 2>/dev/null || true
    fi

    print_status "info" "Cleaning up *.so files..."
    find "$RMSSOURCEDIR" -name "*.so" -type f -delete 2>/dev/null || true

    # Install missing dependencies
    print_header "Installing Missing Dependencies"
    install_missing_dependencies

    print_header "Installing Python Requirements"
    print_status "info" "This may take a few minutes..."
    python -m pip install -r requirements.txt
    print_status "success" "Python requirements installed"

    print_header "Running Setup"
    print_status "info" "Building RMS (this may take a while)..."
    
    # Try build with multiple recovery attempts
    local build_attempts=0
    local max_build_attempts=2
    local build_success=false
    
    while (( build_attempts < max_build_attempts )) && [ "$build_success" = false ]; do
        if python -m pip install -e . --no-deps --no-build-isolation; then
            build_success=true
            break
        else
            ((build_attempts++))
            if (( build_attempts < max_build_attempts )); then
                print_status "warning" "Build failed (attempt $build_attempts/$max_build_attempts). Cleaning and retrying..."
                # Clean build artifacts and try again
                rm -rf build/ dist/ 2>/dev/null || true
                find "$RMSSOURCEDIR" -name "*.pyc" -type f -delete 2>/dev/null || true
                find "$RMSSOURCEDIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
                # Clear pip cache to avoid corrupted wheel issues (pip 21.1+ only)
                if command -v pip >/dev/null 2>&1 && python -m pip cache --help >/dev/null 2>&1; then
                    print_status "info" "Clearing pip cache to avoid wheel corruption..."
                    python -m pip cache purge 2>/dev/null || true
                else
                    print_status "info" "Pip cache command not available, skipping cache clear..."
                fi
                interactive_sleep 2
            fi
        fi
    done
    
    if [ "$build_success" = false ]; then
        print_status "error" "Build failed after $max_build_attempts attempts."
        print_status "warning" "Attempting to restore previous version..."
        
        # Try to rollback to previous commit if available
        if [ -n "${ORIGINAL_COMMIT:-}" ] && [ "$ORIGINAL_COMMIT" != "unknown" ]; then
            print_status "info" "Rolling back to previous commit: $ORIGINAL_COMMIT"
            if git reset --hard "$ORIGINAL_COMMIT" 2>/dev/null; then
                print_status "warning" "Rolled back to previous version. Update failed but system should be functional."
                restore_files || print_status "warning" "Failed to restore config files"
                return
            fi
        fi
        
        print_status "error" "Build recovery failed. Manual intervention may be required."
        print_status "info" "Backup files are available in: $RMSBACKUPDIR"
        exit 1
    fi
    
    print_status "success" "Build completed successfully"
    
    # Print the update report
    print_update_report
    print_status "success" "Update process completed successfully!"
    interactive_sleep 3
}

# Run the main process
main "$@"