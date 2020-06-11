#!/bin/bash

# monitor the logfile and if the capture process stops, restart it
sudo logger 'RMS pi4 watchdog process starting'
hn=`hostname`
MAILRECIP=markmcintyre99@googlemail.com

cd /home/pi/RMS_data/logs
fn=`ls -1 log* | tail -1`

dead="no"
while [ "$dead" != "yes" ]
do
  logf=`find . -name $fn -mmin +1 -ls`
  if [ "$logf" !=  "" ] ; then
    done=`grep Archiving $fn`
    if [ "$done" != "" ] ; then
        sudo logger 'RMS Pi4 watchdog process finished cleanly'
        exit 0
    fi
    touch /home/pi/source/RMS/.crash
    sudo loggger 'RMS Pi4 watchdog RMS stopped acquisition'
    killall python
    sleep 2
    cd ~/source/RMS
    lxterminal -e Scripts/RMS_StartCapture.sh -r

    echo From: pi@${hn} > ./stopped.txt
    echo To: $MAILRECIP >> ./stopped.txt
    echo Subject: process stopped >> ./stopped.txt
    echo stopped at `date +%Y%m%d-%H%M%S` >> ./stopped.txt
    echo $logf >>  ./stopped.txt
    /usr/bin/msmtp -t < ./stopped.txt
    sudo logger 'RMS Pi4 watchdog continuing...'
  fi
  echo sleeping
  sleep 20
done
sudo logger 'watchdog exiting...'

exit 0