**Enabling the watchdog service**

Sometimes the RPi whill hang for unknown reasons, so it is a good idea to restart it then. This can be done automatically using the onboard watchdog timer.

1. Install the watchdog package:

```
sudo apt-get install watchdog
```

2. Load the module manually:

```
sudo modprobe bcm2835_wdt
```

3. Then, add a config to automatically load the module:
Open config file:
```
sudo nano /etc/modules-load.d/bcm2835_wdt.conf
```

Add this line in the file and save it:

```
bcm2835_wdt
```

4. We’ll also need to manually edit the systemd unit at /lib/systemd/system/watchdog.service:

```
sudo nano /lib/systemd/system/watchdog.service
```

and add a line to the [Install] section:

```
[Install]
WantedBy=multi-user.target
```

Also, make sure to add a ' at the end of the line starting with "ExecStartPre=". There is an error in the original file and this has to be fixed, see here: https://github.com/debian-pi/raspbian-ua-netinst/issues/298

5. We need to configure the watchdog.

Open /etc/watchdog.conf with your favorite editor.

```
sudo nano /etc/watchdog.conf
```

Uncomment the line that starts with #watchdog-device by removing the hash (#) to enable the watchdog daemon to use the watchdog device.

Uncomment the line that says #max-load-1 = 24 by removing the hash symbol to reboot the device if the load goes over 24 over 1 minute. Change this number to 100, so it will read:

```
max-load-1 = 100
```

A load of 100 of one minute means that you would have needed 100 Raspberry Pis to complete that task in 1 minute.

Finally, add this line at the end of the file:

```
watchdog-timeout       = 14
```

Values of 15 or higher won't work for some reason...

6. Then enable the service:

```
sudo systemctl enable watchdog.service
```

7. Finally, start the service:

```
sudo systemctl start watchdog.service
```

8. You can set various options for the watchdog in /etc/watchdog.conf – see the man page for that file.
