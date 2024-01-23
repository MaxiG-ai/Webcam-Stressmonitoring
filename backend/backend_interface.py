from abc import ABC
import numpy as np


class BackendInterface(ABC):

    """
    Interface for the backend that is used by the frontend.
    This class is also used to prevent circular imports by using the principle of dependency inversion.
    """
    def get_image(self) -> np.array:
        """
        Get the image from the video feed.

        :return: the image
        """
        pass

    def get_results(self) -> dict:
        """
        Get the heartrate.

        :return: the heartrate
        """
        pass