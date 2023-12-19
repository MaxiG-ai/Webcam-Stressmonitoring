import numpy as np
from PyQt5.QtWidgets import QApplication

from frontend.i_frontend import IFrontend
from frontend.main_window import MainWindow
from frontend.image_stream import ImageStream


class Gui(IFrontend):

    def __init__(self):
        # initialize the gui
        self._application = QApplication([])
        self._main_window = MainWindow()

        # setup image stream and connect it to the gui
        self._image_stream = ImageStream()
        self._image_stream.register_to_QImage_stream(self._main_window.update_image)

    def start(self):
        self._main_window.show()
        self._image_stream.start_QImage_stream()
        self._application.exec_()

    def get_image(self) -> np.ndarray:
        return self._image_stream.get_single_image()

    def set_heartrate(self, heartrate: int):
        self._main_window.set_heartrate(heartrate)
