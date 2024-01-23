from PyQt5.QtWidgets import QWidget, QLabel


class Settings(QLabel):

    def __init__(self, parent: QWidget = None):
        super().__init__(parent=parent)
        self.setText("Settings")