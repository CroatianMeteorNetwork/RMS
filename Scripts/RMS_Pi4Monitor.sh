#!/bin/bash

# monitor the logfile and if the capture process stops, restart it
logger 'RMS pi4 watchdog process starting'
hn=`hostname`

cd /home/pi/RMS_data/logs
dead="no"
while [ "$dead" != "yes" ]
do
  fn=`ls -1tr log* | tail -1`
  logf=`find . -name $fn -mmin +1 -ls`
  dayt=`wc -l $fn | awk '{print $1}'`
  if [[ "$logf" !=  "" && $dayt -gt 30 ]] ; then
    done=`grep Archiving $fn`
    if [ "$done" != "" ] ; then
        logger 'RMS Pi4 watchdog process finished cleanly'
        exit 0
    fi
    logger 'RMS Pi4 watchdog RMS stopped acquisition'
    killall python
    sleep 2
    cd ~/source/RMS
    lxterminal -e Scripts/RMS_StartCapture.sh -r
    cd /home/pi/RMS_data/logs
    logger 'RMS Pi4 watchdog continuing...'
  fi
  cd /home/pi/RMS_data/logs
  acqui=`grep "Starting capture" $fn`
  sleepint=20
  if [ "$acqui" == "" ] ; then
    sleepint=600
  fi
  sleep $sleepint
done
logger 'watchdog exiting...'

exit 0

