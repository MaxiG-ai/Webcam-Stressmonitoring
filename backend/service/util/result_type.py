from enum import Enum


class ResultType(Enum):
    """Enumeration of result types.

    This enum represents a collection result types used by monitoring services.
    """
    HEARTRATE = 1,
    MOVING_AVERAGE_HEARTRATE = 2,
    FRAMERATE = 3,
    BOUNDING_BOX = 4,
    EMOTION = 5
