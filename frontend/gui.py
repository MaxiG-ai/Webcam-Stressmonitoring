from PyQt5.QtWidgets import QApplication

from frontend.window import Window
from frontend.results import Results
from frontend.settings import Settings
from frontend.video import Video


from backend.backend_interface import BackendInterface



class Gui:

    def __init__(self, backend: BackendInterface):
        """
        Create a new GUI.
        """
        self.app = QApplication([])
        self.window = Window(backend)


    def initialize(self):
        """
        """

    def run(self):
        """
        Start the GUI.
        """
        self.window.show()
        self.app.exec_()

    def update_frame(self, frame):
        self.window.update_frame(frame=frame)

    def update_heartrate(self, heartrate):
        """
        Update the GUI with the latest heartrate.
        """
        pass

    def update_moving_average_heartrate(self, moving_average_heartrate):
        """
        Update the GUI with the latest moving average heartrate.
        """
        pass

    def update_framerate(self, framerate):
        """
        Update the GUI with the latest framerate.
        """
        pass

    def update_bounding_boxes(self, bounding_boxes):
        """
        Update the GUI with the latest bounding boxes.
        """
        pass
