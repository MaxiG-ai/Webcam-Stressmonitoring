import numpy as np
import os
import time
import cv2

from backend.backend_interface import BackendInterface
from backend.service.util.result_type import ResultType
from backend.service.video_heartrate.video_heartrate_monitoring_service import VideoHeartrateMonitoringService
from backend.video.video_feed import VideoFeed
from backend.video.video_source import VideoSource
from backend.parameter_settings import MONITORING_CONTROLLER_SETTINGS
from frontend.gui import Gui

from PyQt5.QtCore import QLibraryInfo
# from PySide2.QtCore import QLibraryInfo

# set location of qt plugins
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)


class MonitoringController(BackendInterface):
    """Class for controlling the stress monitoring services program.

    This class controls the different modules used in the stress monitoring
    program. It handles the video feed, controls the monitoring services and
    updates the user interface.
    """
    def __init__(self):
        """
        Create a new monitoring controller.
        """
        self.video_feed = VideoFeed(MONITORING_CONTROLLER_SETTINGS.get_value('VideoSource'))
        self.video_heartrate_monitoring_service = VideoHeartrateMonitoringService(self.video_feed)

    def initialize(self):
        """
        Initialize the monitoring controller.
        """
        print('Started Initialization.')

        self.video_feed.initialize()
        self.video_feed.run()
        time.sleep(1)

        self.video_heartrate_monitoring_service.initialize()
        self.video_heartrate_monitoring_service.run()
        time.sleep(10)

        print('Finished Initialization.')

    def get_image(self) -> np.array:
        """
        Get the image from the video feed.

        :return: the image
        """
        return self.video_feed.get_latest_frame()

    def get_results(self) -> dict:
        """
        Get the heartrate.

        :return: the heartrate
        """
        return self.video_heartrate_monitoring_service._fetch()
