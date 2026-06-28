#!/bin/bash
echo "Starting RMS..."
sleep 10
source ~/vRMS/bin/activate
cd ~/source/RMS

# === MEMPROFILE (debug-memprofile branch only) ===========================
# Enables the in-RMS memory profiler: a whole-box sampler thread + an in-process
# native probe in BufferedCapture that logs glibc mallinfo + the pipeline-rebuild
# count, so RssShmem / child count can be correlated against reconnect churn.
# Output goes to the normal RMS log plus the CSV below. Remove this block to disable.
export RMS_MEMPROFILE=60
export RMS_MEMPROFILE_CSV=~/RMS_data/logs/rms_memprofile.csv
# Do NOT set MALLOC_ARENA_MAX here - it would mask arena retention we want to observe.
# =========================================================================

# Init log file
LOGPATH=~/RMS_data/logs/
LOGDATE=$(date +"%Y%m%d_%H%M%S")
LOGSUFFIX="_log.txt"
LOGFILE=$LOGPATH$LOGDATE$LOGSUFFIX

mkdir -p $LOGPATH

echo ""
echo ""
echo "If you need to update the RMS config file, you can do it now."
echo "Any changes to the config file will be read only after this script is started again or the Pi is rebooted."
echo ""
sleep 5

python -m RMS.StartCapture "$@"

read -p "Press any key to continue... "

$SHELL
