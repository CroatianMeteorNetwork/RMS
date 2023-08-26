#!/bin/sh
cd ~/source/raspberrypi/i2c_cmd/bin
chmod 777 *.sh
sleep 1
./veye_mipi_i2c.sh -w -f videoformat -p1 PAL
sleep 3
./veye_mipi_i2c.sh -w -f sharppen -p1 0
sleep 1
./veye_mipi_i2c.sh -w -f irtrigger -p1 1
sleep 1
./veye_mipi_i2c.sh -w -f lowlight -p1 0  # fixed 25 fps
sleep 1
./veye_mipi_i2c.sh -w -f wdrmode -p1 0  # back light mode off
sleep 1
./veye_mipi_i2c.sh -w -f denoise -p1 0xc
sleep 1
./veye_mipi_i2c.sh -w -f daynightmode -p1 0xfe	# B&W mode
sleep 1
./veye_mipi_i2c.sh -w -f new_expmode -p1 1
sleep 1
./veye_mipi_i2c.sh -w -f new_mshutter -p1 40000 # fixed 40 ms 
sleep 1
./veye_mipi_i2c.sh -w -f new_mgain -p1 20	# 0.1 - 0.3
sleep 1
./veye_mipi_i2c.sh -w -f brightness -p1 0
sleep 1
# special code for sky imaging, to turn automatic bad point correction off
./i2c_write 10 0x3b 0x0007 0xFE
./i2c_write 10 0x3b 0x0010 0xDB
./i2c_write 10 0x3b 0x0011 0x9F
./i2c_write 10 0x3b 0x0012 0x00
./i2c_write 10 0x3b 0x0013 0x00
sleep 1
./i2c_read 10 0x3b 0x0014 1
./veye_mipi_i2c.sh -w -f paramsave
sleep 1
# make sure the params are correct
./veye_mipi_i2c.sh -r -f new_expmode
./veye_mipi_i2c.sh -r -f new_mgain
./veye_mipi_i2c.sh -r -f new_mshutter



