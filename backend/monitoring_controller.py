import time

import cv2

from backend.service.util.result_type import ResultType
from backend.service.video_heartrate.video_heartrate_monitoring_service import VideoHeartrateMonitoringService
from backend.video.video_feed import VideoFeed
from backend.video.video_source import VideoSource


class MonitoringController:
    """Class for controlling the stress monitoring services program.

    This class controls the different modules used in the stress monitoring
    program. It handles the video feed, controls the monitoring services and
    updates the user interface.
    """
    def __init__(self):
        """
        Create a new monitoring controller.
        """
        self.video_feed = VideoFeed(VideoSource.DEMO)
        self.video_heartrate_monitoring_service = VideoHeartrateMonitoringService(self.video_feed)

    def initialize(self):
        """
        Initialize the monitoring controller.
        """
        self.video_feed.initialize()
        self.video_feed.run()
        time.sleep(1)

        self.video_heartrate_monitoring_service.initialize()
        self.video_heartrate_monitoring_service.run()
        time.sleep(4)

    def run(self):
        """
        Start the monitoring controller.
        """
        while True:
            frame = self.video_feed.get_latest_frame()
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
            cv2.imshow('Stress Monitoring', frame)

            print("Moving Average BPM: {:.2f}, BPM: {:.2f}, FS: {:.2f}".format(
                moving_average_heartrate,
                heartrate,
                framerate
            ))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
