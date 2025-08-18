#!/bin/bash

# Exit immediately as Istrastream is not longer used
exit 0

echo ""
echo "START EXTERNAL SCRIPT..." 
echo ""
echo "CHECKING ARGUMENTS..."
echo ""
if [[ -z "$1" || -z "$2" || -z "$3" || -z "$4" || -z "$5" || -z "$6" || -z "$7" || -z "$8" || -z "$9" ]]; then
	echo "MISSING ARGUMENTS..."
	echo "EXIT..."
	echo ""	
	exit
fi


echo "CHECKING DEPENDENCIES"
if [[ ( $( command -v avconv ) || $( command -v ffmpeg ) ) && $( command -v convert ) ]]; then
        echo "ALL DEPENDENCIES ALREADY INSTALLED!"
else
	if $( sudo -n true ) ; then
        echo "INSTALLING DEPENDENCIES..."
        sudo apt-get update
        sudo apt-get -y install imagemagick 
        sudo apt-get -y install libav-tools
        sudo apt-get -y install ffmpeg
	sudo apt-get -y install curl
    else
    	echo "NO SUDO PRIVILEDGES TO INSTALL DEPENDENCIES..."
    fi
fi


# If avconv is not available, set up an alias for ffmpeg
if [ ! $( command -v avconv ) ]; then
	alias avconv="ffmpeg"
fi


###SETUP###
SYSTEM="rms"
STATION_ID=$1
CAPTURED_DIR_NAME=$2
ARCHIVED_DIR_NAME=$3
LATITUDE=$4
LONGITUDE=$5
ELEVATION=$6
WIDTH=$7
HEIGHT=$8
REMAINING_SECONDS=$9

echo "REMAININS SECONDS = $REMAINING_SECONDS"
echo ""

LEVEL_1_SECONDS=3600 # 60 min => all files
LEVEL_2_SECONDS=1800 # 30 min => without captured_stack
LEVEL_3_SECONDS=300 # 5 min => without captured_stack and without timelapse_video
LEVEL_1=false
LEVEL_2=false
LEVEL_3=false

if [ "$REMAINING_SECONDS" -lt "$LEVEL_2_SECONDS" ]; then
	LEVEL_3=true
	ACTIVE_LEVEL=3
elif [ "$REMAINING_SECONDS" -lt "$LEVEL_1_SECONDS" ]; then
	LEVEL_2=true
	ACTIVE_LEVEL=2
elif [ "$REMAINING_SECONDS" -ge "$LEVEL_1_SECONDS" ]; then
	LEVEL_1=true
	ACTIVE_LEVEL=1
fi

echo "ACTIVE_LEVEL = $ACTIVE_LEVEL"
echo ""

DATE_NOW=$(date +"%Y%m%d")
TMP_VIDEO_FILE="$CAPTURED_DIR_NAME/$(basename $CAPTURED_DIR_NAME).mp4"
VIDEO_FILE="$CAPTURED_DIR_NAME/${STATION_ID}_${DATE_NOW}.mp4"
FTP_SERVER="server.istrastream.com"
SERVER="http://istrastream.com"
AGENT="$SYSTEM-$STATION_ID"
FTP_DETECT_INFO="$(ls $CAPTURED_DIR_NAME/*.txt | grep 'FTPdetectinfo' | grep -vE "uncalibrated" | grep -vE "unfiltered")"

#Thanks to Alfredo Dal'Ava Junior :)
if [[ -z "$FTP_DETECT_INFO" ]]; then
   METEOR_COUNT="0"
else
   METEOR_COUNT="$(sed -n 1p $FTP_DETECT_INFO | awk '{ print $NF }')"
fi

function generate_timelapse {
	SECONDS_LIMIT=$(expr $REMAINING_SECONDS - 120)
	if ($LEVEL_1); then
		cd ~/source/RMS
		python -m Utils.GenerateTimelapse $CAPTURED_DIR_NAME
		mv $TMP_VIDEO_FILE $VIDEO_FILE
	fi
	if ($LEVEL_2); then
		cd ~/source/RMS
		python -m Utils.GenerateTimelapse $CAPTURED_DIR_NAME
		mv $TMP_VIDEO_FILE $VIDEO_FILE
	fi
}

function upload_info {				
	curl "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=info&meteor_count=$METEOR_COUNT&action=info&level=$ACTIVE_LEVEL"	
}

function upload_captured_stack {
	CAPTURED_STACK_FILE="$(find $CAPTURED_DIR_NAME -name '*.jpg' | grep '_captured_stack.')"
	TMP_CAPTURED_STACK_FILE="$CAPTURED_DIR_NAME/captured_stacck.jpg"
	if [ -f "$CAPTURED_STACK_FILE" ]; then
		convert $CAPTURED_STACK_FILE $TMP_CAPTURED_STACK_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$TMP_CAPTURED_STACK_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=captured_stack&meteor_count=$METEOR_COUNT&action=upload&level=$ACTIVE_LEVEL"
	else			
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_captured_stack&meteor_count=$METEOR_COUNT&action=upload&level=$ACTIVE_LEVEL"
	fi	
}

function upload_stack {	
	STACK_FILE="$(ls $CAPTURED_DIR_NAME/*.jpg | grep '_stack_')"
	if [ -f "$STACK_FILE" ]; then			
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$STACK_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=stack&meteor_count=$METEOR_COUNT&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_stack&meteor_count=$METEOR_COUNT&action=upload&level=$ACTIVE_LEVEL"
	fi
}

function upload_captured {	
	CAPTURED_FILE="$(ls $CAPTURED_DIR_NAME/*.jpg | grep 'CAPTURED')"
	if [ -f "$CAPTURED_FILE" ]; then				
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$CAPTURED_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=captured&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_captured&action=upload&level=$ACTIVE_LEVEL"
	fi
}

function upload_detected {	
	DETECTED_FILE="$(ls $CAPTURED_DIR_NAME/*.jpg | grep 'DETECTED')"
	if [ -f "$DETECTED_FILE" ]; then			
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$DETECTED_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=detected&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_detected&action=upload&level=$ACTIVE_LEVEL"
	fi
}

function upload_fieldsums_noavg {	
	FIELDSUMS_NOAVG_FILE="$(ls $CAPTURED_DIR_NAME/*.png | grep 'fieldsums_noavg.png')"
	TMP_FIELDSUMS_NOAVG_FILE="$CAPTURED_DIR_NAME/fieldsums_noavg.jpg"
	if [ -f "$FIELDSUMS_NOAVG_FILE" ]; then
		convert $FIELDSUMS_NOAVG_FILE $TMP_FIELDSUMS_NOAVG_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$TMP_FIELDSUMS_NOAVG_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=fieldsums_noavg&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_fieldsums_noavg&action=upload&level=$ACTIVE_LEVEL"
	fi
}

function upload_fieldsums {
	FIELDSUMS_FILE="$(ls $CAPTURED_DIR_NAME/*.png | grep 'fieldsums.png')"
	TMP_FIELDSUMS_FILE="$CAPTURED_DIR_NAME/fieldsums.jpg"
	if [ -f "$FIELDSUMS_FILE" ]; then
		convert $FIELDSUMS_FILE $TMP_FIELDSUMS_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$TMP_FIELDSUMS_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=fieldsums&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_fieldsums&action=upload&level=$ACTIVE_LEVEL"
	fi
}

function upload_report_astrometry {
	REPORT_ASTROMETRY_FILE="$(ls $CAPTURED_DIR_NAME/*.jpg | grep 'report_astrometry.jpg')"
	TMP_REPORT_ASTROMETRY_FILE="$CAPTURED_DIR_NAME/report_astrometry.jpg"
	if [ -f "$REPORT_ASTROMETRY_FILE" ]; then
		cp $REPORT_ASTROMETRY_FILE $TMP_REPORT_ASTROMETRY_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$REPORT_ASTROMETRY_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=report_astrometry&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_report_astrometry&action=upload&level=$ACTIVE_LEVEL"
	fi
}

function upload_report_photometry {
	REPORT_PHOTOMETRY_FILE="$(ls $CAPTURED_DIR_NAME/*.png | grep 'report_photometry.png')"
	TMP_REPORT_PHOTOMETRY_FILE="$CAPTURED_DIR_NAME/report_photometry.jpg"
	if [ -f "$REPORT_PHOTOMETRY_FILE" ]; then
		convert $REPORT_PHOTOMETRY_FILE $TMP_REPORT_PHOTOMETRY_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$TMP_REPORT_PHOTOMETRY_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=report_photometry&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_report_photometry&action=upload&level=$ACTIVE_LEVEL"
	fi
}

function upload_calibration_variation {
	CALIBRATION_VARIATION_FILE="$(find $CAPTURED_DIR_NAME -name '*.png' | grep 'calibration_variation.png')"
	TMP_CALIBRATION_VARIATION_FILE="$CAPTURED_DIR_NAME/calibration_variation.jpg"
	if [ -f "$CALIBRATION_VARIATION_FILE" ]; then
		convert $CALIBRATION_VARIATION_FILE $TMP_CALIBRATION_VARIATION_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$TMP_CALIBRATION_VARIATION_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=calibration_variation&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_calibration_variation&action=upload&level=$ACTIVE_LEVEL"
	fi	
}

function upload_radiants {
	RADIANTS_FILE="$(find $CAPTURED_DIR_NAME -name '*.png' | grep 'radiants.png')"
	TMP_RADIANTS_FILE="$CAPTURED_DIR_NAME/radiants.jpg"
	if [ -f "$RADIANTS_FILE" ]; then
		convert $RADIANTS_FILE $TMP_RADIANTS_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$TMP_RADIANTS_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=radiants&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_radiants&action=upload&level=$ACTIVE_LEVEL"
	fi	
}

function upload_photometry_variation {
	PHOTOMETRY_VARIATION_FILE="$(find $CAPTURED_DIR_NAME -name '*.png' | grep 'photometry_variation.png')"
	TMP_PHOTOMETRY_VARIATION_FILE="$CAPTURED_DIR_NAME/photometry_variation.jpg"
	if [ -f "$PHOTOMETRY_VARIATION_FILE" ]; then
		convert $PHOTOMETRY_VARIATION_FILE $TMP_PHOTOMETRY_VARIATION_FILE
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$TMP_PHOTOMETRY_VARIATION_FILE" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=photometry_variation&action=upload&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_photometry_variation&action=upload&level=$ACTIVE_LEVEL"
	fi	
}

function update_calib_report_photometry_status {
	CALIB_REPORT_PHOTOMETRY_FILE="$(find $CAPTURED_DIR_NAME -name '*.png' | grep 'calib_report_photometry.png')"	
	if [ -f "$CALIB_REPORT_PHOTOMETRY_FILE" ]; then 	
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=calib_report_photometry&action=update&level=$ACTIVE_LEVEL"
	else
		curl --user-agent $AGENT "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=no_calib_report_photometry&action=update&level=$ACTIVE_LEVEL"
	fi	
}

function upload_video_file {	
	curl --user-agent $AGENT -F"operation=upload" -F"file=@$VIDEO_FILE" "http://server.istrastream.com/?station_id=$STATION_ID&system=$SYSTEM&action=upload"	
}

function upload_kml_file {
	KML_FILE_100="$CAPTURED_DIR_NAME/$STATION_ID-100km.kml"	
	if [ -f "$KML_FILE_100" ]; then		
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$KML_FILE_100" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=kml100&action=upload&level=$ACTIVE_LEVEL"	
	fi	
	KML_FILE_70="$CAPTURED_DIR_NAME/$STATION_ID-70km.kml"	
	if [ -f "$KML_FILE_70" ]; then		
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$KML_FILE_70" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=kml70&action=upload&level=$ACTIVE_LEVEL"	
	fi	
	KML_FILE_25="$CAPTURED_DIR_NAME/$STATION_ID-25km.kml"	
	if [ -f "$KML_FILE_25" ]; then		
		curl --user-agent $AGENT -F"operation=upload" -F"file=@$KML_FILE_25" "$SERVER/$SYSTEM/?station_id=$STATION_ID&type=kml25&action=upload&level=$ACTIVE_LEVEL"	
	fi	
}

VAR_1="1"
VAR_2="1"

if [ $VAR_1 = $VAR_2 ]; then	
	
	# echo "GENERATE TIMELAPSE VIDEO..."
	# generate_timelapse
	# echo ""
	
	echo "UPLOAD INFO..."
	upload_info
	echo ""

	echo "UPLOAD CAPTURED STACK..."
	upload_captured_stack
	echo ""

	echo "UPLOAD STACK..."
	upload_stack
	echo ""

	echo "UPLOAD CAPTURED THUMB..."
	upload_captured
	echo ""

	echo "UPLOAD DETECTED THUMB..."
	upload_detected
	echo ""

	echo "UPLOAD FIELDSUMS_NOAVG..."
	upload_fieldsums_noavg
	echo ""

	echo "UPLOAD FIELDSUMS..."
	upload_fieldsums
	echo ""

	echo "UPLOAD REPORT ASTROMETRY..."
	upload_report_astrometry
	echo ""

	echo "UPLOAD REPORT PHOTOMETRY..."
	upload_report_photometry
	echo ""

	echo "UPLOAD CALIBRATION VARIATION..."
	upload_calibration_variation
	echo ""
	
	echo "UPLOAD RADIANTS..."
	upload_radiants
	echo ""

	echo "UPLOAD PHOTOMETRY VARIATION..."
	upload_photometry_variation
	echo ""

	echo "UPDATE CALIB REPORT PHOTOMETRY STATUS..."
	update_calib_report_photometry_status
	echo ""
	
	echo "UPLOAD KML FILE..."
	upload_kml_file
	echo ""

	if [ -e "$VIDEO_FILE" ]; then
    	echo "UPLOADING VIDEO TO $FTP_SERVER..."
		upload_video_file			
		sleep 5			
		echo ""
	else
		echo "VIDEO FILE FAIL..."
		echo ""
	fi 
fi
echo "END EXTERNAL SCRIPT..."
echo ""
