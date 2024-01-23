from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2

from backend.backend_interface import BackendInterface

FPS = 30  # Adjust as needed


class Video(QWidget):
    def __init__(self, parent: QWidget, backend: BackendInterface, **kwargs):
        super().__init__(parent=parent, **kwargs)

        self.backend = backend
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.image_label)

        self.q_image = None
        self.pix_map = QPixmap()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_frame)

    def start_updating(self):
        self.update_timer.start(int(1000 / FPS))  # Timer interval in milliseconds

    def _update_frame(self):
        frame = self.backend.get_image()
        self.q_image = self._convert_to_QImage(frame)
        self.pix_map = QPixmap.fromImage(self.q_image)
        self.image_label.setPixmap(self.pix_map)

    @staticmethod
    def _convert_to_QImage(cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return q_image