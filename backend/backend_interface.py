from abc import ABC


class BackendInterface(ABC):

    """
    Interface for the backend that is used by the frontend.
    This class is also used to prevent circular imports by using the principle of dependency inversion.
    """