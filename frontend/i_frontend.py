from abc import abstractmethod
import numpy as np


class IFrontend:

    @abstractmethod
    def get_image(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_heartrate(self, heartrate: int):
        pass
