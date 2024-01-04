from enum import Enum


class EyeBoxStrategy(Enum):
    """Enumeration of eye box strategies.

    This enum represents a collection of various eye box strategies used in the
    video heartrate monitoring service.
    """
    STRICT = 1
    CONTRACTION = 2
    APPROXIMATION = 3
    DISABLED = 4
