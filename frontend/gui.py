from PyQt5.QtWidgets import QApplication



class Gui:

    def __init__(self):
        """
        Create a new GUI.
        """
        self.app = QApplication([])

    def initialize(self):
        """
        Initialize the GUI.
        """
        print('Started Initialization.')

        print('Finished Initialization.')

    def run(self):
        """
        Start the GUI.
        """
        self.app.exec_()

    def update_frame(self, frame):
        """
        Update the GUI with the latest frame.
        """
        pass

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
