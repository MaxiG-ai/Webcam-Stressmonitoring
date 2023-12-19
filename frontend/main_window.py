from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout


HEARTRATE_LABEL_TEXT = "Heartrate: {} bpm"


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        # create the main layout
        self._main_layout = QHBoxLayout(self)

        # create the image label and add it to the main layout
        self._image_label = QLabel(self)
        self._image_label.setScaledContents(True)
        self._main_layout.addWidget(self._image_label)

        # create the sidebar and add it to the main layout
        self._sidebar = QWidget(self)
        self._sidebar.setFixedWidth(200)
        self._sidebar_layout = QVBoxLayout(self._sidebar)

        self._heartrate_label = QLabel(text=HEARTRATE_LABEL_TEXT.format(0))
        self._sidebar_layout.addWidget(self._heartrate_label)

        self._main_layout.addWidget(self._sidebar)

    def update_image(self, image: QImage):
        self._image_label.setPixmap(QPixmap.fromImage(image))

    def set_heartrate(self, heartrate: int):
        self._heartrate_label.setText(HEARTRATE_LABEL_TEXT.format(heartrate))