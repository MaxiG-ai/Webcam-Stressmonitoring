import warnings

from abc import ABC, abstractmethod

from backend.parameter_settings import MONITORING_SERVICE_SETTINGS


class MonitoringService(ABC):
    """Abstract class that represents a monitoring service.

    This abstract class provides methods that need to be implemented by a
    derived class for it to be used as a monitoring service in the architecture.

    The following methods need to be implemented by derived classes:
        - _initialize()
        - _run()
        - _halt()
        - _fetch()
    """
    def __init__(self, name='Monitoring', result_types=None, settings=MONITORING_SERVICE_SETTINGS):
        """
        Create a monitoring service.

        :param name: the name of the service
        :param result_types: the result types as a list of ResultType enums
        :param settings: the settings of the service as a ParameterSettings instance
        """
        if result_types is None:
            result_types = []
        self.name = name
        self.result_types = result_types
        self.settings = settings
        self.is_initialized = False
        self.is_running = False

    def get_name(self):
        """
        Get the name of the service

        :return: the name
        """
        return self.name

    def get_result_types(self):
        """
        Get the result types of the service.

        :return: the result types
        """
        return self.result_types

    def get_settings(self):
        """
        Get the settings of the service

        :return: the settings
        """
        return self.settings

    def is_initialized(self):
        """
        Returns whether the service has been initialized or not.

        :return: True if the service was initialized, False otherwise
        """
        return self.is_initialized

    def is_running(self):
        """
        Returns whether the service is running or not.

        :return: True if the service is running, False otherwise
        """
        return self.is_running

    def initialize(self):
        """
        Initialize the service.
        """
        self.is_initialized = self._initialize()

    def run(self):
        """
        Start the service.
        """
        self._validate_initialization()
        self._validate(not self.is_running, 'The service "' + self.name + '" is already running.')
        if self._run():
            self.is_running = True
        else:
            warnings.warn('The service "' + self.name + '" was not started properly.', UserWarning)

    def halt(self):
        """
        Stop the service.
        """
        self._validate_initialization()
        self._validate(self.is_running, 'The service "' + self.name + '" is already halted.')
        if self._halt():
            self.is_running = False
        else:
            warnings.warn('The service "' + self.name + '" was not stopped properly.', UserWarning)

    def fetch(self):
        """
        Fetch the current results of this service.

        :return: a dictionary containing the results
        """
        self._validate_initialization()
        self._validate(self.is_running, 'The service "' + self.name + '"is not running.')
        return self._fetch()

    @staticmethod
    def _validate(condition, warning_message):
        """
        Validate whether the condition is true and print a warning message if it is not.

        :param condition: the condition
        :param warning_message: the warning message
        """
        if not condition:
            warnings.warn(warning_message, UserWarning)

    def _validate_initialization(self):
        """
        Validate whether the service is initialized and print a warning message if it is not.
        """
        self._validate(self.is_initialized, 'The service "' + self.name + '" was not initialized properly.')

    @abstractmethod
    def _initialize(self):
        """
        Initialize the service. Needs to be implemented by derived classes.

        :return: True if initializing the service was successful, False otherwise
        """
        pass

    @abstractmethod
    def _run(self):
        """
        Start the service. Needs to be implemented by derived classes.

        :return: True if the starting the service was successful, False otherwise
        """
        pass

    @abstractmethod
    def _halt(self):
        """
        Stop the service. Needs to be implemented by derived classes.

        :return: True if stopping the service was successful, False otherwise
        """
        pass

    @abstractmethod
    def _fetch(self):
        """
        Fetch the current results of this service. Needs to be implemented by derived classes.

        :return: a dictionary containing the results
        """
        pass
