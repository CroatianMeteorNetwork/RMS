#!/bin/bash

# This script handles the post-update tasks after git operations are complete.
# It's called by RMS_Update.sh after the "danger zone" to ensure we're running
# the latest version of the post-update logic.

# Directories and variables (passed as environment variables from RMS_Update.sh)
RMSSOURCEDIR="${RMSSOURCEDIR:-~/source/RMS}"
RMSBACKUPDIR="${RMSBACKUPDIR:-~/.rms_backup}"
UPDATEINPROGRESSFILE="${UPDATEINPROGRESSFILE:-$RMSBACKUPDIR/update_in_progress}"
RMS_BRANCH="${RMS_BRANCH}"
INITIAL_BRANCH="${INITIAL_BRANCH}"
INITIAL_COMMIT="${INITIAL_COMMIT}"
INITIAL_DATE="${INITIAL_DATE}"
RMS_REMOTE="${RMS_REMOTE}"

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

# Install missing dependencies
install_missing_dependencies() {
    local SYSTEM_PACKAGES="$RMSSOURCEDIR/system_packages.txt"
    
    if [ ! -f "$SYSTEM_PACKAGES" ]; then
        echo "Warning: System packages file not found: $SYSTEM_PACKAGES"
        return
    fi

    local missing_packages=()

    # -----------------------------------------------------------------------------
    # We store system-level dependencies in a separate file (system_packages.txt)
    # so that when RMS_PostUpdate pulls new code (including a potentially updated list of packages),
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
        # Clear screen and show prominent message for sudo
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
    
    tput bold
    echo "Branch update summary:"
    echo "  From: $INITIAL_BRANCH (${INITIAL_COMMIT} - ${INITIAL_DATE})"
    echo "  To:   $RMS_BRANCH (${final_commit} - ${final_date})"
    tput sgr0
}

#######################################################
######################   MAIN   #######################
#######################################################

# Change to the RMS source directory
cd "$RMSSOURCEDIR" || { print_status "error" "RMS source directory not found. Exiting."; exit 1; }

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