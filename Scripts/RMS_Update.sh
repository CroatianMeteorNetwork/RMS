#!/bin/bash

# This script is used for updating the RMS code from GitHub

# WARNING: The update might fail when new dependencies (libraires)
#  are introduced! Further steps might have to be undertaken.


RMSSOURCEDIR=~/source/RMS

RMSBACKUPDIR=~/.rms_backup

echo "Updating RMS code..."

# If an argument (any) is given, then the config and mask won't be backed up
if [ $# -eq 0 ]; then
    
    echo "Backing up the config and mask..."

    # Make the backup directory
    mkdir $RMSBACKUPDIR

    # Back up the config and the mask
    cp $RMSSOURCEDIR/.config $RMSBACKUPDIR/.
    cp $RMSSOURCEDIR/mask.bmp $RMSBACKUPDIR/.
fi


cd $RMSSOURCEDIR

# Remove the build dir
rm -r build

# Pull new code from github
git stash
git pull

# Activate the virtual environment
source ~/vRMS/bin/activate

# Run the python setup
python setup.py install


if [ $# -eq 0 ]; then
    
    # Copy the config and the mask back
    cp $RMSBACKUPDIR/.config $RMSSOURCEDIR/.
    cp $RMSBACKUPDIR/mask.bmp $RMSSOURCEDIR/.
fi
