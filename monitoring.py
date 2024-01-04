from backend.monitoring_controller import MonitoringController

if __name__ == "__main__":
    controller = MonitoringController()
    controller.initialize()
    controller.run()