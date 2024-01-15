import numpy as np
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from backend.backend_interface import BackendInterface

from frontend.video import Video
from frontend.settings import Settings
from frontend.results import Results


INITIAL_WINDOW_WIDTH = 1280
INITIAL_WINDOW_HEIGHT = 720
WINDOW_TITLE = r"Webcam-Stress-Monitoring"


class Window(QWidget):

    def __init__(self,  backend: BackendInterface):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
        self.backend = backend

        self.video: QWidget = Video()
        self.settings: QWidget = Settings()
        self.results: QWidget = Results()

        # define horizontal boxes for video and sidebar
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.addWidget(self.video)

        # define sidebar with results and settings and add to horizontal layout
        self.sidebar = QWidget()
        self.vertical_layout = QVBoxLayout(self.sidebar)
        self.horizontal_layout.addWidget(self.vertical_layout)

    def update_frame(self, frame: np.array):
        self.video.update_frame(frame)
