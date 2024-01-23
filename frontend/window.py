from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from backend.backend_interface import BackendInterface

from frontend.video import Video
from frontend.results import Results
from frontend.settings import Settings
from .util.css_styles import MAIN_STYLE_SHEET


INITIAL_WINDOW_WIDTH = 1280
INITIAL_WINDOW_HEIGHT = 720
WINDOW_TITLE = r"Webcam-Stress-Monitoring"


class Window(QWidget):

    def __init__(self,  backend: BackendInterface):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
        self.setStyleSheet(MAIN_STYLE_SHEET)

        self.video = Video(parent=self, backend=backend)
        self.sidebar_widget = QWidget()
        self.main_layout = QHBoxLayout(self)
        self.main_layout.addWidget(self.video, stretch=2)
        self.main_layout.addWidget(self.sidebar_widget, stretch=1)
        self.setLayout(self.main_layout)

        self.results = Results(parent=self.sidebar_widget, backend=backend)
        self.settings = Settings(parent=self.sidebar_widget)
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.addWidget(self.results)
        self.sidebar_layout.addWidget(self.settings)
        self.sidebar_widget.setLayout(self.sidebar_layout)

    def start_updating(self):
        self.video.start_updating()
        self.results.start_updating()