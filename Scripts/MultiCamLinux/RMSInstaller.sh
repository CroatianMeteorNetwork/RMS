#
# This is a modified version of the standard RMS on Linux installer linked to on the wiki.
# It includes custom commands to automate the Multi station gui enabled software
sudo apt-get update
sudo apt-get -y upgrade
if grep -Fq Debian /etc/issue; then
# ..might need to fix  for Bullseye..
sudo apt-get install -y python3-tk python3-pil
wget http://ftp.br.debian.org/debian/pool/main/x/xcb-util/libxcb-util1_0.4.0-1+b1_amd64.deb
sudo dpkg -i libxcb-util1_0.4.0-1+b1_amd64.deb
elif grep -Fq 'Ubuntu 20.04' /etc/issue; then
sudo apt-get install -y python3.8-tk libxslt-dev python-imaging-tk
elif grep -Fq 'Ubuntu 22.04' /etc/issue; then
sudo apt-get install -y python3-tk libxslt1-dev python3-pil
fi
sudo apt-get install -y git mplayer python3 python3-dev python3-pip libblas-dev libatlas-base-dev \
liblapack-dev at-spi2-core libopencv-dev libffi-dev libssl-dev socat ntp \
libxml2-dev libxslt-dev imagemagick ffmpeg cmake unzip

sudo pip3 install --upgrade pip
sudo apt install python3-virtualenv
cd ~
virtualenv vRMS
source ~/vRMS/bin/activate
pip install -U pip setuptools
pip install numpy==1.23.5
pip install Pillow
pip install gitpython cython pyephem astropy 
pip install scipy==1.8.1
pip install paramiko==2.8.0
pip install matplotlib
pip install imreg_dft
pip install configparser==4.0.2
pip install imageio==2.6.1
pip install pyfits
pip install PyQt5
pip install pyqtgraph

cd ~/source/RMS
cp  ~/source/RMS/Scripts/MultiCamLinux/opencv4_install.sh .
./opencv4_install.sh ~/vRMS
cd ~/source
sudo apt install -y gstreamer1.0*
sudo apt install -y gstreamer1.0-python3-dbg-plugin-loader
sudo apt install -y gstreamer1.0-python3-plugin-loader
sudo apt install -y ubuntu-restricted-extras
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev


git clone https://github.com/opencv/opencv.git
cd opencv/
git checkout 4.1.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D BUILD_EXAMPLES=ON ..
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON
sudo make -j$(nproc)
sudo make install
sudo ldconfig

cd ~/source/RMS
python setup.py install
sudo apt install -y gstreamer1.0-plugins-good python3-pyqt5
# get CMNbinViewer....
cd ~/source
git clone https://github.com/CroatianMeteorNetwork/cmn_binviewer.git
# check to see if a desktop is installed - not foolproof - doesnt check for X11 env etc..
if [ -d "$HOME/Desktop" ]
then 
# generate desktop links
cd ~/source/RMS/Scripts
./GenerateDesktopLinks.sh
fi
