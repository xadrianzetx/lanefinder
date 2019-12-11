# lanefinder

[![Build Status](https://travis-ci.org/xadrianzetx/lanefinder.svg?branch=master)](https://travis-ci.org/xadrianzetx/lanefinder)

TPU accelerated traffic lane segmentation engine for your Raspberry Pi!

<p align="center">
  <img src="https://github.com/xadrianzetx/lanefinder/blob/master/assets/gifs/videofeed.gif?raw=true">
</p>

Thanks to combined power of Raspberry Pi and Edge TPU, this lane segmentation engine is small enough to actually fit in car as regular dashcam, efficient enough to run from powerbank and fast enough to provide real-time traffic lane detection support in low visibility conditions.

## Models

Currently lanefinder runs on Unet with MobileNetV2 backbone and custom decoder. Model has been trained on [CU Lane Dataset](https://xingangpan.github.io/projects/CULane.html) and finetuned using only night time images collected during some long autumn nights. In order to run on edgetpu, frozen graph went through process of [full integer quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations). I've published code allowing you to train this net [here](https://github.com/xadrianzetx/mobileunet-tensorflow). Future releases should include model selection and improved performance (that is if I manage to cram EfficientNet on TPU).

## Framerate

At the moment lanefinder supports two modes - camera feed and video playback. Former holds steady ~30 fps while latter tends to drop to around 10 fps due to some unknown at this point video/acceleration issue. Didn't look into it yet since playback is not the main (nor default) mode and I only implemented it to record prototype.

## Prototype

<p align="center">
  <img src="https://github.com/xadrianzetx/lanefinder/blob/master/assets/gifs/prototype.gif?raw=true">
</p>

## Hardware requirements

* RaspberryPi 3 B+ running Raspbian Stretch (support for Buster in future release)
* Some kind of touchscreen for Pi
* Pi camera (don't go cheap here, the bigger FOV the better)
* Powerbank capable of at least 2.4A output, if you want to test this in your car (as i did)
* [Edge TPU](https://coral.ai/products/accelerator)

## Install and Run

Easy. Use included setup script by running `sudo chmod +x install.sh && ./install.sh`. Remeber to reenable camera interface using `sudo raspi-config` after reboot! After this you are all set up and can start lanefinder with simple `python3 main.py`. Current build supports Raspbian Stretch with Python 3.5 If TPU has not been detected, lanefinder runs in passthrough mode since inference on Pi processor only is not supported.