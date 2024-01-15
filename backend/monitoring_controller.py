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
        self.gui = Gui(backend=self)

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

        self.gui.initialize()
        self.gui.run()

        print('Finished Initialization.')

    def run(self):
        """
        Start the monitoring controller.
        """
        while True:
            frame = np.copy(self.video_feed.get_latest_frame())
            video_heartrate = self.video_heartrate_monitoring_service.fetch()

            heartrate = video_heartrate[ResultType.HEARTRATE]
            moving_average_heartrate = video_heartrate[ResultType.MOVING_AVERAGE_HEARTRATE]
            framerate = video_heartrate[ResultType.FRAMERATE]
            bounding_boxes = video_heartrate[ResultType.BOUNDING_BOX]

            for ((box_left, box_top, box_right, box_bottom), style) in bounding_boxes:
                frame = cv2.rectangle(
                    frame,
                    (box_left, box_top),
                    (box_right, box_bottom),
                    **style
                )

            self.gui.update_frame(frame)
            self.gui.update_heartrate(heartrate)
            self.gui.update_moving_average_heartrate(moving_average_heartrate)
            self.gui.update_framerate(framerate)
            self.gui.update_bounding_boxes(bounding_boxes)

            if cv2.waitKey(int(1000 / MONITORING_CONTROLLER_SETTINGS.get_value('FPS'))) & 0xFF == ord('q'):
                break
