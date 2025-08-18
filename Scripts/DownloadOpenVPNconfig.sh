#!/bin/bash

# Server address
HOST=gmn.uwo.ca

# TLS version string to add to OpenVPN file if it's missing
TLS_STRING="tls-version-min 1.0"

# Path to the OpenVPN config file
CONFIG_PATH=/etc/openvpn/client.conf


# Take the username as the argument
#if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
if [ $# -ne 1 ]; then
  echo "Usage: $0 USER_NAME" >&2
  exit 1
fi

# Convert the input to lowercase
USNAME="${1,,}"



# Install OpenVPN
sudo apt-get update
sudo apt-get install openvpn -y

# Copy the VPN configuration from the server
sftp -o StrictHostKeyChecking=no $USNAME@$HOST:files/openvpn/$USNAME.ovpn ~/$USNAME.ovpn

# Install the configuration
sudo cp ~/$USNAME.ovpn $CONFIG_PATH

# Add the TLS minimum version if it't not in the config file
if ! grep -q "$TLS_STRING" $CONFIG_PATH; then

  sudo sed "/^verb.*/i $TLS_STRING" $CONFIG_PATH --in-place=".bak"
  echo "Inserted: $TLS_STRING"

fi


# Automatically start the VPN service
sudo /etc/init.d/openvpn start

# Test connectivity
sudo openvpn --client --config /etc/openvpn/client.conf

$SHELL