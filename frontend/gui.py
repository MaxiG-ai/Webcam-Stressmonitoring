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

    def run(self):
        """
        Start the GUI.
        """
        self.window.show()
        self.window.start_updating()
        self.app.exec_()