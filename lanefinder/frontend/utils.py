import numpy as numpy


def rgba_to_kivy(channels):
    """
    Kivy color converter

    Produces RGBA scaled to [0, 1] interval
    background image must be disabled before use

    :param channel: list
                    list of RGBA channel values

    :return:        list
                    scaled RGBA channels
    """
    scaled = [channel / 255 for channel in channels[:-1]]

    return scaled + [channels[-1]]
