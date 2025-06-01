#!/bin/bash

# Configuration variables
CAMERA_SUBNET="192.168.42"
RPI_LAST_OCTET="51"          # Change this for each RPi (51, 52, 53, etc)
NETWORK_NAME="ipcams"        # Name of the network connection

# Derived variables
RPI_IP="${CAMERA_SUBNET}.${RPI_LAST_OCTET}/24"
SUBNET_IP="${CAMERA_SUBNET}.1/24"

# Function to find the ethernet interface
find_ethernet_interface() {
    # Try to find a physical ethernet interface
    local interfaces=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^(eth|eno|enp)')
    
    if [ -z "$interfaces" ]; then
        echo "No ethernet interface found!"
        exit 1
    fi
    
    # If multiple interfaces found, let user choose
    if [ $(echo "$interfaces" | wc -l) -gt 1 ]; then
        echo "Multiple ethernet interfaces found:"
        local i=1
        while IFS= read -r interface; do
            echo "$i) $interface"
            i=$((i+1))
        done <<< "$interfaces"
        
        read -p "Select interface number: " selection
        INTERFACE=$(echo "$interfaces" | sed -n "${selection}p")
    else
        INTERFACE=$interfaces
    fi
    
    echo "Using ethernet interface: $INTERFACE"
}

# Function to find the WiFi connection
find_wifi_connection() {
    # Get active WiFi connection
    local wifi_con=$(nmcli -t -f NAME,DEVICE,TYPE connection show --active | grep -E 'wl|wifi' | cut -d':' -f1)
    
    if [ -z "$wifi_con" ]; then
        echo "No active WiFi connection found!"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        WIFI_CON=""
    else
        WIFI_CON=$wifi_con
        echo "Found WiFi connection: $WIFI_CON"
    fi
}

# Function to clean up existing connections
cleanup_existing_connections() {
    echo "Cleaning up existing camera network configurations..."
    
    # Get list of connections with the target name
    local existing_connections=$(nmcli -t -f NAME,UUID connection show | grep "^${NETWORK_NAME}:" | cut -d':' -f2)
    
    if [ ! -z "$existing_connections" ]; then
        echo "Found existing camera network configurations. Removing..."
        while IFS= read -r uuid; do
            echo "Removing connection with UUID: $uuid"
            nmcli connection delete uuid "$uuid"
        done <<< "$existing_connections"
    fi
    
    # Also try to delete by name (backup method)
    nmcli connection delete "$NETWORK_NAME" 2>/dev/null || true
}

# Function to check if NetworkManager is installed and running
check_network_manager() {
    echo "Checking NetworkManager installation..."
    if ! command -v nmcli &> /dev/null; then
        echo "NetworkManager not found. Installing..."
        sudo apt update && sudo apt install -y network-manager
    fi
    
    echo "Enabling and starting NetworkManager..."
    sudo systemctl enable NetworkManager
    sudo systemctl start NetworkManager
}

# Function to configure the network
configure_network() {
    echo "Configuring network..."
    
    # Clean up existing configurations first
    cleanup_existing_connections
    
    # Create ethernet connection for cameras
    echo "Creating ethernet connection '${NETWORK_NAME}'..."
    sudo nmcli connection add \
        con-name "${NETWORK_NAME}" \
        ifname "${INTERFACE}" \
        type ethernet \
        ip4 "${SUBNET_IP}"

    # Set RPi's ethernet IP
    echo "Setting RPi IP to ${RPI_IP}..."
    sudo nmcli connection modify "${NETWORK_NAME}" \
        ipv4.method manual \
        ipv4.addresses "${RPI_IP}"

    # Bring up the connection
    echo "Bringing up connection..."
    sudo nmcli connection up "${NETWORK_NAME}"

    # Configure network priorities
    echo "Configuring network priorities..."
    if [ ! -z "$WIFI_CON" ]; then
        echo "Setting WiFi ($WIFI_CON) priority..."
        sudo nmcli connection modify "$WIFI_CON" \
            ipv4.route-metric 100 \
            ipv4.method auto \
            connection.autoconnect yes \
            connection.autoconnect-priority 999
        
        # Re-activate WiFi connection to apply metrics
        sudo nmcli connection down "$WIFI_CON"
        sudo nmcli connection up "$WIFI_CON"
    fi

    echo "Setting ethernet priority..."
    sudo nmcli connection modify "${NETWORK_NAME}" \
        ipv4.route-metric 600 \
        connection.autoconnect yes \
        connection.autoconnect-priority 1
    
    # Bring connections down/up to ensure metrics are applied
    sudo nmcli connection down "${NETWORK_NAME}"
    sudo nmcli connection up "${NETWORK_NAME}"

    echo "Restarting NetworkManager..."
    sudo systemctl restart NetworkManager
    
    # Wait for network to stabilize
    sleep 2
    
    # Verify metrics
    echo "Verifying route metrics..."
    ip route show | grep -E "metric|via"
}

# Function to verify network setup
verify_network() {
    echo -e "\nVerifying network configuration..."
    
    # Check WiFi status
    echo -e "\nWiFi Status (Internet Connection):"
    ip addr show wlp3s0
    echo -e "\nInternet Routing:"
    ip route get 8.8.8.8
    
    # Check Camera Network Status
    echo -e "\nCamera Network Status (Ethernet):"
    ip addr show ${INTERFACE}
    echo -e "\nCamera Network Routes:"
    ip route show | grep "192.168.42"
    
    echo -e "\nNetwork Manager Connections:"
    nmcli connection show
    
    # Add status summary
    echo -e "\nNetwork Status Summary:"
    echo "Internet: Connected via WiFi (wlp3s0)"
    echo "Camera Network: Ready on ${INTERFACE} (will activate when cable connected)"
    echo "Camera Subnet: 192.168.42.0/24"
    echo "This RPi's Camera Network IP: 192.168.42.${RPI_LAST_OCTET}"
}

# Function to display camera IP information
show_camera_info() {
    echo -e "\nCamera IP Configuration Guide:"
    echo "Configure your cameras with these IP addresses:"
    echo "Camera 1: ${CAMERA_SUBNET}.101"
    echo "Camera 2: ${CAMERA_SUBNET}.102"
    echo "Camera 3: ${CAMERA_SUBNET}.103"
    echo "(Continue this pattern for additional cameras)"
    echo -e "\nMake sure each camera has netmask 255.255.255.0"
}

# Main execution
echo "Starting network configuration for IP cameras..."

# Find interfaces
find_ethernet_interface
find_wifi_connection

echo "RPi will be configured with IP: ${RPI_IP}"

# Ask for confirmation
read -p "Continue with network configuration? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    check_network_manager
    configure_network
    verify_network
    show_camera_info
    echo -e "\nConfiguration complete!"
else
    echo "Configuration cancelled."
    exit 1
fi