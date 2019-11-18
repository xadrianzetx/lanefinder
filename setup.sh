#!/bin/bash

# touchscreen setup
sudo apt-get install -y xinput-calibrator

git clone https://github.com/waveshare/LCD-show.git && \
cd LCD-show/

# reboots pi after this op
sudo chmod +x LCD35-show && \
./LCD35-show
