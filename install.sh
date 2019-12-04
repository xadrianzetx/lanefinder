#!/bin/bash

sudo apt-get update

# install opencv and edgetpu libs
sudo apt-get -y install libglib2.0-0 libsm6 libxext6 libxrender-dev python3-edgetpu libedgetpu1-std

# install python requirements
wget https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl
pip3 install -r requirements.txt