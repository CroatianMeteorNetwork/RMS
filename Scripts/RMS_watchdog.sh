#!/bin/bash

# Monitor the kernel log for memory alloc errors and restart RMS if they occur
tail -Fn0 /var/log/kern.log | \
  while read line ; do
    echo "$line" | grep "alloc failed"
    if [ $? = 0 ]
    then
      sudo logger 'alloc failed - watchdog triggered'
      touch /home/pi/source/RMS/.crash
      
      #sudo reboot

      # Restart RMS
      killall python
      sleep 2
      cd ~/source/RMS
      lxterminal -e Scripts/RMS_StartCapture.sh -r
    fi

    sudo logger 'watchdog continuing...'

  done

  sudo logger 'watchdog exiting...'

exit 0