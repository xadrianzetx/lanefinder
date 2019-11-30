import os
import cv2
import yaml
from lanefinder import Lanefinder


def read_config():
    if not os.path.isfile('config.yaml'):
        raise FileNotFoundError('Could not find config file')

    with open('config.yaml', 'r') as file:
        config = yaml.load(file)

    return config


def main():
    # set video stream to fullscreen
    window_name = 'lanefinder'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    config = read_config()

    lanefinder = Lanefinder(
        model=config['model'],
        input_shape=config['input_shape'],
        output_shape=tuple(config['output_shape']),
        quant=config['quantization'],
        dequant=config['dequantization']
    )

    # set window name to one with fullscren property
    # and run
    lanefinder.window = window_name
    lanefinder.stream()
    lanefinder.destroy()


if __name__ == '__main__':
    main()
