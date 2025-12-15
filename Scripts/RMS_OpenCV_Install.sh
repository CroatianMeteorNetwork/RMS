#!/bin/bash

# Check if the virtual environment path is provided
if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 VIRTUALENV_PATH" >&2
  exit 1
fi

# Configuration
HOME_DIR=$HOME
OPENCV_VERSION=4.11.0
VIRTUALENV_PATH=$1

# Automatically locate Python executable and version
PYTHON_EXECUTABLE=$(which python3)
PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")

# Detect CPU architecture
CPU_ARCH=$(uname -m)

# Update and upgrade the system
sudo apt-get update
sudo apt-get upgrade -y

# Install common dependencies
# Use the system-installed TBB (libtbb-dev) instead of building the bundled version
sudo apt-get install -y  libtbb-dev
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y libgtk-3-dev libatlas-base-dev gfortran

# Install additional dependencies for ARM (Raspberry Pi)
if [[ "$CPU_ARCH" == "armv7l" || "$CPU_ARCH" == "aarch64" ]]; then
  sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-omx
  sudo apt-get install -y libharfbuzz-dev libfribidi-dev libilmbase-dev libopenexr-dev
fi

# Architecture-specific optimizations
NEON_OPT=""
INTEL_OPT=""
MAKE_JOBS=$(nproc)

if [[ "$CPU_ARCH" == "armv7l" || "$CPU_ARCH" == "aarch64" ]]; then
  # Optimizations for Raspberry Pi or ARM-based systems
  NEON_OPT="-D ENABLE_NEON=ON -D ENABLE_VFPV4=ON"
  MAKE_JOBS=$(($(nproc) / 2))  # Limit jobs for low-memory devices
elif [[ "$CPU_ARCH" == "x86_64" ]]; then
  # Optimizations for Intel/AMD systems
  INTEL_OPT="-D WITH_OPENMP=ON"
fi

# Download and unpack OpenCV
cd ${HOME_DIR}
wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
unzip -o opencv.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
unzip -o opencv_contrib.zip

# Build and install OpenCV
cd ${HOME_DIR}/opencv-${OPENCV_VERSION}/
mkdir -p build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${VIRTUALENV_PATH} \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=${HOME_DIR}/opencv_contrib-${OPENCV_VERSION}/modules \
    -D PYTHON_DEFAULT_EXECUTABLE=${PYTHON_EXECUTABLE} \
    -D PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
    -D WITH_opencv_python3=ON \
    -D WITH_TIFF=ON \
    -D BUILD_TBB=OFF \
    -D WITH_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=OFF \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_VTK=OFF \
    -D OPENCV_EXTRA_EXE_LINKER_FLAGS=-latomic \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=TRUE \
    -D WITH_GSTREAMER=ON \
    -D WITH_FFMPEG=ON \
    $NEON_OPT \
    $INTEL_OPT ..

make -j${MAKE_JOBS}
sudo make install
sudo ldconfig

# Clean up
cd ${HOME_DIR}
rm -rf opencv-${OPENCV_VERSION} opencv_contrib-${OPENCV_VERSION} opencv.zip opencv_contrib.zip
