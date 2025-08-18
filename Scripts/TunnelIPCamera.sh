#!/bin/bash
echo "Tunneling camera ports to RPi public ports..."
socat tcp-listen:34567,reuseaddr,fork tcp:192.168.42.10:34567 &
socat tcp-listen:8899,reuseaddr,fork tcp:192.168.42.10:8899 &
sudo socat tcp-listen:80,reuseaddr,fork tcp:192.168.42.10:80 &
sudo socat tcp-listen:554,reuseaddr,fork tcp:192.168.42.10:554 &
sudo socat UDP4-RECVFROM:554,reuseaddr,fork UDP4-SENDTO:192.168.42.10:554 &

read -p "Press any key to stop tunnelling..."
killall socat
$SHELL