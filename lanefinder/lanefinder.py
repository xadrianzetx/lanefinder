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

    def _postprocess(self, frame):
        pass

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
            pred = self._engine.RunInference(frame.flatten())

            if self._window is not None:
                # show in window with fullscreen setup
                cv2.imshow(self._window, frmcpy)

            else:
                # user did not specify window name
                # for fullscreen use
                cv2.imshow('default', frmcpy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # exit on key press
                break

    def destroy(self):
        """
        """
        cv2.destroyAllWindows()
        self._cap.release()
