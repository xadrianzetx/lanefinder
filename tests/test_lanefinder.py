import os
import yaml
import unittest
import numpy as np
from lanefinder import Lanefinder


class TestLanefinder(unittest.TestCase):

    @staticmethod
    def read_config():
        if not os.path.isfile('config.yaml'):
            raise FileNotFoundError('Could not find config file')

        with open('config.yaml', 'r') as file:
            config = yaml.load(file)

        return config

    def test_preprocess_out_dtype(self):
        """
        Test data type after preprocessing step
        """
        config = self.read_config()
        lanefinder = Lanefinder(
            model=config['model'],
            input_shape=config['input_shape'],
            output_shape=tuple(config['output_shape']),
            quant=config['quantization'],
            dequant=config['dequantization']
        )

        frame = np.zeros(shape=(config['input_shape'] + [3]), dtype=np.float32)
        out = lanefinder._preprocess(frame)
        self.assertTrue(out.dtype.name, 'uint8')

    def test_preprocess_out_shape(self):
        """
        Test data shape after postprocessing step
        """
        config = self.read_config()
        lanefinder = Lanefinder(
            model=config['model'],
            input_shape=config['input_shape'],
            output_shape=tuple(config['output_shape']),
            quant=config['quantization'],
            dequant=config['dequantization']
        )

        frame = np.zeros(shape=(config['input_shape'] + [3]), dtype=np.float32)
        out = lanefinder._preprocess(frame)
        fshape = [1] + config['input_shape'] + [3]
        self.assertEquals(list(out.shape), fshape)

    def test_preprocess_out_framesize(self):
        """
        Test if frame size has desired dimensions after preprocessing
        """
        config = self.read_config()
        lanefinder = Lanefinder(
            model=config['model'],
            input_shape=config['input_shape'],
            output_shape=tuple(config['output_shape']),
            quant=config['quantization'],
            dequant=config['dequantization']
        )

        frame = np.zeros(shape=(config['input_shape'] + [3]), dtype=np.float32)
        out = lanefinder._preprocess(frame)
        self.assertEquals(list(out[0, :, :, 0].shape), config['input_shape'])

    def test_postprocess_out_dtype(self):
        """
        Test frame dtype after postprocessing
        """
        config = self.read_config()
        lanefinder = Lanefinder(
            model=config['model'],
            input_shape=config['input_shape'],
            output_shape=tuple(config['output_shape']),
            quant=config['quantization'],
            dequant=config['dequantization']
        )

        frame = np.zeros(shape=(config['input_shape'] + [3]), dtype=np.float32)
        mask = np.zeros(shape=(192 * 192,), dtype=np.uint8)
        predobj = [0, mask]
        out = lanefinder._postprocess(predobj, frame)
        self.assertTrue(out.dtype.name, 'float32')

    def test_postprocess_out_framesize(self):
        """
        Test frame shape after postprocessing
        """
        config = self.read_config()
        lanefinder = Lanefinder(
            model=config['model'],
            input_shape=config['input_shape'],
            output_shape=tuple(config['output_shape']),
            quant=config['quantization'],
            dequant=config['dequantization']
        )

        frame = np.zeros(shape=(config['input_shape'] + [3]), dtype=np.float32)
        mask = np.zeros(shape=(192 * 192,), dtype=np.uint8)
        predobj = [0, mask]
        out = lanefinder._postprocess(predobj, frame)
        fshape = (config['output_shape'][1], config['output_shape'][0], 3)
        self.assertEquals(out.shape, fshape)


if __name__ == '__main__':
    unittest.main()
