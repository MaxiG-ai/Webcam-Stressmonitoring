from enum import Enum


class DataType(Enum):
    """Enumeration of basic data types.

    This enum represents a collection of basic data types used in the application.
    """
    INTEGER = 1,
    FLOAT = 2,
    BOOLEAN = 3
    CATEGORICAL = 4
