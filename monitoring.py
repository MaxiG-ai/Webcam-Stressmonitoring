from backend.monitoring_controller import MonitoringController
from frontend.gui import Gui

if __name__ == "__main__":
    controller = MonitoringController()
    controller.initialize()

    gui = Gui(backend=controller)
    gui.run()