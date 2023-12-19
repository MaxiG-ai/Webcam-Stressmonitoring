import time
import cv2
from PyQt5.QtGui import QImage
from threading import Thread
import numpy as np
from typing import List


FPS = 60


class ImageStream:

    def __init__(self):
        self._video_capture = cv2.VideoCapture(0)

        # thread that will
        self._stream_feeder = Thread(target=self._feed_QImage_stream)
        self._stream_feeder.daemon = True

        # list of methods that will be called with the QImage as parameter when a new image is available
        self._observers: List[callable] = []

    def get_single_image(self) -> np.ndarray:
        _, frame = self._video_capture.read()
        return frame

    def register_to_QImage_stream(self, observer: callable):
        self._observers.append(observer)

    def start_QImage_stream(self):
        self._stream_feeder.start()

    def _feed_QImage_stream(self):
        while True:
            _, frame = self._video_capture.read()
            image = self._convert_to_QImage(frame)
            self.distribute_QImage(image)
            time.sleep(1 / FPS)

    @staticmethod
    def _convert_to_QImage(cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format

    def distribute_QImage(self, image: QImage):
        for observer in self._observers:
            observer(image)