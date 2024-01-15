import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap



class Video(QWidget):

    def __init__(self, parent: QWidget = None):
        super().__init__(parent=parent)

        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.image_label)

    def update_frame(self, frame: np.array):
        q_image = self._convert_to_QImage(frame)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    @staticmethod
    def _convert_to_QImage(cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format
