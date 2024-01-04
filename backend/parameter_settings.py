from backend.util.data_type import DataType
from backend.service.video_heartrate.eye_box_strategy import EyeBoxStrategy


class ParameterSettings:
    """Class containing parameter information.

    This class contains parameter metadata, consisting of possible values and
    a default value.
    """
    def __init__(self, parameter_metadata=None):
        """
        Create a new parameter settings instance.

        :param parameter_metadata: a dictionary whose entries have the form
                                   "name: (list of possible values / datatype, default value)"
        """
        if parameter_metadata is None:
            parameter_metadata = {}
        self.parameter_metadata = parameter_metadata
        self.parameter_values = {
            parameter: parameter_metadata[parameter][1] for parameter in parameter_metadata
        }

    def get_parameters(self):
        """
        Get the parameters.

        :return: the parameters
        """
        return list(self.parameter_metadata.keys())

    def get_possible_values(self, parameter):
        """
        Get the possible values of a parameter.

        :param parameter: the possible values
        :return: the possible values of the parameter
        """
        return self.parameter_metadata[parameter][0]

    def get_default_value(self, parameter):
        """
        Get the default value of a parameter.

        :param parameter: the default value
        :return: the default value of the parameter
        """
        return self.parameter_metadata[parameter][1]

    def get_value(self, parameter):
        """
        Get the current value of a parameter.

        :param parameter: the parameter
        :return: the current value of the parameter
        """
        return self.parameter_values[parameter]

    def set_value(self, parameter, value):
        """
        Set the current value of a parameter.

        :param parameter: the parameter
        :param value: the value
        """
        self.parameter_values[parameter] = value


VIDEO_FEED_SETTINGS = ParameterSettings({
    'ScalingFactor': (DataType.FLOAT, 0.4),
    'LoopingEnabled': (DataType.BOOLEAN, True)
})

MONITORING_SERVICE_SETTINGS = ParameterSettings({})

VIDEO_HEARTRATE_MONITORING_SERVICE_SETTINGS = ParameterSettings({
    'EyeBoxStrategy': ([
        EyeBoxStrategy.STRICT,
        EyeBoxStrategy.CONTRACTION,
        EyeBoxStrategy.APPROXIMATION,
        EyeBoxStrategy.DISABLED
    ], EyeBoxStrategy.CONTRACTION),
    'EyeBoxHorizontalContractionFactor': (DataType.FLOAT, 0.1),
    'EyeBoxVerticalContractionFactor': (DataType.FLOAT, 0.25),
    'EyeBoxApproximationRatio': (DataType.FLOAT, 0.4),
    # 'ImageScalingFactor': (DataType.FLOAT, 0.4, None),
    # 'ImageScalingFactor': (DataType.FLOAT, 1.0, None),
    'IntensitiesWindowSize': (DataType.INTEGER, 120),
    'MovingAverageHeartrateWindowSize': (DataType.INTEGER, 512),
    'BoundingBoxUpdateRate': (DataType.FLOAT, 0.4),
    'DisplayFaceBox': (DataType.BOOLEAN, True),
    'DisplayRoiBox': (DataType.BOOLEAN, True),
    'DisplayLeftEyeBox': (DataType.BOOLEAN, True),
    'DisplayRightEyeBox': (DataType.BOOLEAN, True)
})
