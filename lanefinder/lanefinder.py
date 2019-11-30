import cv2
import numpy as np
from edgetpu.basic.basic_engine import BasicEngine


class Lanefinder:

    def __init__(self, model, input_shape, output_shape, quant, dequant):
        self._window = None
        self._engine = self._get_tpu_engine(model)
        self._cap = cv2.VideoCapture(0)
        self._size = input_shape
        self._output_shape = output_shape
        self._quant = quant
        self._dequant = dequant

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, name):
        self._window = name

    @staticmethod
    def _get_tpu_engine(model):
        try:
            # get runtime for TPU
            model = BasicEngine(model)

        except RuntimeError:
            # TPU has not been detected
            model = None

        return model

    def _preprocess(self, frame):
        # normalize and quantize input
        # with paramaeters obtained during
        # model calibration
        frame *= (1 / 255)
        expd = np.expand_dims(frame, axis=0)
        quantized = (expd / self._quant['std'] + self._quant['mean'])

        return quantized.astype(np.uint8)

    def _postprocess(self, pred_obj, frame):
        # get predicted mask in shape (n_rows*n_cols, )
        # and reshape back to (n_rows, n_cols)
        pred = pred_obj[1].reshape(self._size)

        # dequantize and cast back to float
        dequantized = (self._dequant['std'] * (pred - self._dequant['mean']))
        dequantized = dequantized.astype(np.float32)

        # resize frame and mask to output shape
        frame = cv2.resize(frame, self._output_shape)
        mask = cv2.resize(dequantized, (frame.shape[1], frame.shape[0]))
        
        # perform closing operation on mask to smooth out lane edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # overlay frame and segmentation mask
        frame[mask != 0] = (255, 0, 255)

        return frame

    def stream(self):
        """
        Starts real time video stream with
        coral edgetpu supported traffic lane segmentation

        :return:    void
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

            if self._engine is not None:
                # TPU engine has been initiated
                # so run inference steps
                frame = self._preprocess(frame)
                pred_obj = self._engine.RunInference(frame.flatten())
                pred = self._postprocess(pred_obj, frmcpy)

            else:
                # no TPU detected so output recorded
                # frame with warning sign on it
                height, width, _ = frmcpy.shape
                pred = cv2.putText(
                    frmcpy,
                    'TPU has not been detected!',
                    org=(height // 2, width // 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255),
                    thickness=cv2.LINE_AA
                )

            if self._window is not None:
                # show in window with fullscreen setup
                cv2.imshow(self._window, pred)

            else:
                # user did not specify window name
                # for fullscreen use so use default opencv size
                cv2.imshow('default', pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # exit on key press
                break

    def destroy(self):
        """
        Runs cleanup after main loop exit

        :return:    void
        """
        cv2.destroyAllWindows()
        self._cap.release()


class LanefinderFromVideo(Lanefinder):

    def __init__(self, src, model, input_shape, quant, dequant):
        Lanefinder.__init__(self, model, input_shape, quant, dequant)
        self._cap = cv2.VideoCapture(src)
