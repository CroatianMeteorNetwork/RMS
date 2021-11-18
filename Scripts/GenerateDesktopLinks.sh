# Get location of the script

if [ -L $0 ] ; then
    DIR=$(dirname $(readlink -f $0)) ;
else
    DIR=$(dirname $0) ;
fi ;

DIR=$(realpath $DIR)


# Get full path of Desktop
DESKTOP=$(realpath ~/Desktop)

# Create links to scripts, so they can be updated
ln -sf $DIR/CMNbinViewer_env.sh $DESKTOP/CMNbinViewer.sh
ln -sf $DIR/DownloadOpenVPNconfig.sh $DESKTOP/.
ln -sf $DIR/RMS_FirstRun.sh $DESKTOP/.
ln -sf $DIR/RMS_ShowLiveStream.sh $DESKTOP/.
ln -sf $DIR/RMS_StartCapture.sh $DESKTOP/.
ln -sf $DIR/RMS_Update.sh $DESKTOP/.
cp $DIR/TunnelIPCamera.sh $DESKTOP/.


