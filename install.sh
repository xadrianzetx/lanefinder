#!/bin/bash

# install opencv and edgetpu libs
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install libglib2.0-0 libsm6 libxext6 libxrender-dev \
python3-edgetpu libedgetpu1-std libcblas-dev \
libhdf5-dev libhdf5-serial-dev libatlas-base-dev \
libjasper-dev  libqtgui4  libqt4-test

# install python requirements
wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl
pip3 install -r requirements.txt

# fix numpy installation
pip3 uninstall numpy

# enable picamera drivers and reboot
echo "sudo modprobe bcm2835-v4l2" >> ~/.profile
sudo reboot