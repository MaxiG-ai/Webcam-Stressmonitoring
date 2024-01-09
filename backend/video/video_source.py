import os

from enum import Enum


class VideoSource(Enum):
    """Enumeration of video sources.

    This enum represents a collection of possible video sources.
    """
    WEBCAM = 1
    FILE = 2
    DEMO = 3


DEMO_FILE_PATH = os.path.join(os.getcwd(), 'data', 'heartrate', 'face_1.mp4')
