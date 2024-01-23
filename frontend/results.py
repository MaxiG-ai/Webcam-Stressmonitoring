from PyQt5.QtWidgets import QWidget, QLabel, QFormLayout
from PyQt5.QtCore import QTimer, Qt

from backend.backend_interface import BackendInterface

RESULT_UPDATES_PER_SECOND = 1


class Results(QWidget):
    def __init__(self, backend: BackendInterface, parent=None):
        super().__init__(parent)

        self.backend = backend
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_results)

        self.headline_label = QLabel("Results", self)
        self.headline_label.setAlignment(Qt.AlignCenter)

        self.tags = [
            QLabel("Heart Rate:"),
            QLabel("Heart Rate Moving Average:"),
        ]

        self.results = [
            QLabel(),
            QLabel(),
        ]

        self.layout = QFormLayout(self)
        self.layout.addRow(self.headline_label)
        for tag, result in zip(self.tags, self.results):
            self.layout.addRow(tag, result)
        self.setLayout(self.layout)

    def set_headline(self, headline):
        self.headline_label.setText(headline)

    def start_updating(self):
        self.update_timer.start(int(1000 / RESULT_UPDATES_PER_SECOND))

    def _update_results(self):
        results = self.backend.get_results()
        for result, label in zip(results.values(), self.results):
            if isinstance(result, float):
                result = round(result, 2)
            label.setText(str(result))