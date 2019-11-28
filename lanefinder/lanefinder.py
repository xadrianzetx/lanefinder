import cv2
import numpy as np
from edgetpu.basic.basic_engine import BasicEngine


class Lanefinder:

    def __init__(self, model, input_shape, quant, dequant):
        self._window = None
        self._engine = BasicEngine(model)
        self._cap = cv2.VideoCapture(0)
        self._size = input_shape
        self._quant = quant
        self._dequant = dequant

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, name):
        self._window = name

    def _preprocess(self, frame):
        frame *= (1 / 255)
        expd = np.expand_dims(frame, axis=0)
        quantized = (expd / self._quant['std'] + self._quant['mean'])

        return quantized.astype(np.uint8)

    def _postprocess(self, pred_obj, frame):
        pred = pred_obj[1].reshape(self._size)
        dequantized = (self._dequant['std'] * (pred - self._dequant['mean']))
        dequantized = dequantized.astype(np.float32)
        mask = cv2.resize(dequantized, (frame.shape[1], frame.shape[0]))
        frame[mask != 0] = (255, 0, 255)

        return frame

    def stream(self):
        """
        """
        while True:
            # get next video frame
            ret, frame = self._cap.read()

            if not ret:
                # frame has not been
                # retrieved
                break

            frame = np.array(frame)
            frmcpy = frame.copy()

            frame = cv2.resize(frame, tuple(self._size))
            frame = frame.astype(np.float32)
            frame = self._preprocess(frame)
            pred_obj = self._engine.RunInference(frame.flatten())
            pred = self._postprocess(pred_obj, frmcpy)

            if self._window is not None:
                # show in window with fullscreen setup
                cv2.imshow(self._window, pred)
                # print(pred[1].reshape(192, 192).shape)

            else:
                # user did not specify window name
                # for fullscreen use
                cv2.imshow('default', pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # exit on key press
                break

    def destroy(self):
        """
        """
        cv2.destroyAllWindows()
        self._cap.release()


class LanefinderFromVideo(Lanefinder):

    def __init__(self, src, model, input_shape, quant, dequant):
        Lanefinder.__init__(self, model, input_shape, quant, dequant)
        self._cap = cv2.VideoCapture(src)
