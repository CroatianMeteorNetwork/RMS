#!/bin/bash

FILEPATH="/etc/openvpn/client.conf"
FILEPATHBAK="/etc/openvpn/client.conf.gmnbak"
TLSLINE="tls-version-min 1.0"
SEDCOMMAND="/^<ca>/ i ${TLSLINE}"
GMNHOST="gmn.uwo.ca"
PISCIDIP="129.100.40.167"

# Make a backup of the original file
sudo -n cp -n "${FILEPATH}" "${FILEPATHBAK}"

# Replace hardcoded IP address with gmn.uwo.ca
sudo -n sed -i "s/${PISCIDIP}/${GMNHOST}/g" "${FILEPATH}"

# If in a GMN VPN config file, find <ca> in the ovpn file and add the tls line if it's missing
grep -q "${GMNHOST}" "${FILEPATH}" && grep -q -e "${TLSLINE}" "${FILEPATH}" || sudo -n sed -i "${SEDCOMMAND}" "${FILEPATH}"