#!/bin/bash

# kivy dependencies
sudo apt update
sudo apt install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
   pkg-config libgl1-mesa-dev libgles2-mesa-dev \
   python-setuptools libgstreamer1.0-dev git-core \
   gstreamer1.0-plugins-{bad,base,good,ugly} \
   gstreamer1.0-{omx,alsa} python-dev libmtdev-dev \
   xclip xsel libjpeg-dev

# python dependencies
pip3 install --upgrade --user setuptools
pip3 install --upgrade --user Cython==0.29.10 pillow
pip3 install -r requirements.txt

echo "done!"