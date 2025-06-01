# Raspberry Pi GPS + Chrony Setup (USB GPS and PPS on GPIO)

This guide walks you through configuring a USB GPS and (optionally) a PPS signal on a Raspberry Pi running Chrony for accurate timekeeping.

---

## 1. Prerequisites

- **Raspberry Pi** (any recent model)  
- **USB GPS module** (appears as `/dev/ttyACM0` or `/dev/ttyUSB0`)  
- *(Optional)* **PPS signal** wired to a GPIO pin (e.g., GPIO18 with `/dev/pps0`)  
- *(Optional)* Internet connectivity if you want fallback NTP sources

---

## 2. Install Required Packages

On Raspberry Pi OS (Debian-based), run:
```bash
sudo apt-get update
sudo apt-get install gpsd chrony minicom pps-tools
```
- **gpsd**: Daemon for reading GPS data  
- **chrony**: Modern NTP service  
- **minicom** (optional): Quick testing of raw GPS data  
- **pps-tools** (optional): Check PPS signal with `ppstest`

---

## 3. Verify GPS

1. **Plug in the USB GPS.**  
2. **Identify the device** (usually `/dev/ttyACM0` or `/dev/ttyUSB0`):
   ```bash
   ls /dev/ttyA* /dev/ttyU*
   ```
   If you see `/dev/ttyACM0`, that’s likely the GPS.

3. **Confirm NMEA data** (optional):
   ```bash
   sudo apt-get install minicom
   sudo minicom -D /dev/ttyACM0 -b 9600
   ```
   You should see lines starting with `$GPRMC`, `$GPGGA`, etc.

---

## 4. (Optional) Configure PPS on GPIO

If your GPS module outputs a PPS signal to a Pi GPIO (commonly GPIO18):

1. **Load the PPS overlay**  
   Edit `/boot/config.txt` and add:
   ```
   dtoverlay=pps-gpio,gpiopin=18
   ```
2. **Reboot** to apply:
   ```bash
   sudo reboot
   ```
3. **Check PPS**:
   ```bash
   ls /dev/pps0
   sudo ppstest /dev/pps0
   ```
   If you see PPS timing info, it’s working.

---

## 5. Manually Test gpsd

1. **Stop existing gpsd**:
   ```bash
   sudo systemctl stop gpsd.socket
   sudo systemctl stop gpsd
   sudo killall gpsd
   ```
2. **Launch gpsd in foreground**:
   ```bash
   sudo gpsd -N -n -D 5 -S 2947 /dev/ttyACM0
   ```
   - `-N`: stay in foreground  
   - `-n`: read data immediately  
   - `-D 5`: debug output  
   - `-S 2947`: enable SHM export & listen on port 2947  

3. **Check fix** in another terminal:
   ```bash
   cgps
   ```
   If you see fix data, gpsd is working.

Press **Ctrl+C** to stop gpsd after testing.

---

## 6. Auto-Start gpsd on Boot

Make systemd point gpsd at the correct device:

```bash
sudo systemctl edit gpsd
```
Add:

```
[Service]
ExecStart=
ExecStart=/usr/sbin/gpsd -n -D 3 -S 2947 /dev/ttyACM0
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable gpsd
sudo systemctl start gpsd
```

---

## 7. Configure Chrony

Edit `/etc/chrony/chrony.conf` (or a file under `/etc/chrony/conf.d/`) to include lines referencing the raw NMEA SHM and (optionally) PPS if available. For example:

```bash
# Use official NTP servers for fallback
pool 2.debian.pool.ntp.org iburst

# NMEA data from gpsd (index 0)
refclock SHM 0 refid GPS delay 0.2 offset 0.0 poll 3 precision 1e-1 trust

# PPS from /dev/pps0 (lock to 'GPS' and prefer)
refclock PPS /dev/pps0 refid GPPS lock GPS prefer poll 3
```

- **`delay 0.2`**: Approx USB latency (tweak as needed)  
- **`trust`**: Tells Chrony to trust local refclock sources  
- **`lock GPS`**: Ties PPS to the same GPS source  
- **`prefer`**: If multiple sources exist, prefer PPS

Restart Chrony:

```bash
sudo systemctl restart chrony
```

---

## 8. Verify Chrony

```bash
chronyc sources -v
```
You might see something like:

```
MS Name/IP address Stratum Poll Reach LastRx Last sample
========================================================
#? GPS                  0   3   377     4   +0.000002s
* GPPS                 0   3   377     4   -0.000000s
```

- `*GPPS` means Chrony selected PPS as the primary time source.  
- `GPS` might show `?` or `-` because PPS is more accurate. That’s normal.

---

## 9. Enjoy Accurate Time

Your Pi should now have a stable time source via GPS + PPS. If something isn’t working, check:

1. **Correct device** (`/dev/ttyACM0` vs. `/dev/ttyUSB0`).  
2. **NMEA data** (via `cgps`).  
3. **PPS** signals (via `ppstest`).  
4. **Chrony config** (proper SHM index and PPS).

Done!