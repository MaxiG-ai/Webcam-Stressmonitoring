import numpy as np


class MovingAverage:
    """Class for calculating the moving average.

    This class is used to calculate the moving average. The most recent values
    are stored and the moving average is determined, depending on the chosen
    window size.
    """
    def __init__(self, initial_value=0, window_size=256):
        """
        Initialize the moving average class with an initial value and a window size.

        :param initial_value: the initial value
        :param window_size: the window size
        """
        self.window_size = window_size
        self.history = np.full(window_size, initial_value)

    def add(self, value):
        """
        Remove the first entry from the history and add the new value at its end.

        :param value: the value
        """
        self.history = np.append(self.history[1:], value)

    def get(self):
        """
        Get the current moving average.

        :return: the moving average
        """
        return np.mean(self.history[-self.window_size:])
