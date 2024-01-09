class CoordinateUtils:
    """Class containing coordinate utility methods.

    This class contains static utility methods used when dealing with coordinates.
    """

    @staticmethod
    def convert_coordinates(coordinates):
        """
        Converts coordinates of the form (left, top, width, height) to coordinates
        of the form (left, top, right, bottom).

        :param coordinates: the coordinates as a tuple consisting of left, top, width, height
        :return: the coordinates as a tuple consisting of left, top, right, bottom
        """
        left, top, width, height = coordinates
        return left, top, left + width, top + height

    @staticmethod
    def update_coordinates(old_coordinates, new_coordinates, update_rate=1.0):
        """
        Updates the coordinates according to a given update rate. A rate of 1
        means that the new coordinates replace the old ones.

        :param old_coordinates: the old coordinates
        :param new_coordinates: the new coordinates
        :param update_rate: the update rate
        :return: the updated coordinates
        """
        old_left, old_top, old_right, old_bottom = old_coordinates
        new_left, new_top, new_right, new_bottom = new_coordinates

        updated_left = int(CoordinateUtils.get_weighted_average(new_left, old_left, update_rate))
        updated_top = int(CoordinateUtils.get_weighted_average(new_top, old_top, update_rate))
        updated_right = int(CoordinateUtils.get_weighted_average(new_right, old_right, update_rate))
        updated_bottom = int(CoordinateUtils.get_weighted_average(new_bottom, old_bottom, update_rate))

        return updated_left, updated_top, updated_right, updated_bottom

    @staticmethod
    def get_weighted_average(x, y, ratio):
        """
        Calculates the weighted average between two values according to a given ratio.

        :param x: the first value
        :param y: the second value
        :param ratio: the ratio
        :return: the weighted average
        """
        return x * ratio + y * (1 - ratio)

    @staticmethod
    def order_eyes(eyes):
        """
        Orders eyes by comparing their left coordinate. The one with the smaller
        left coordinate is the left eye.

        :param eyes: a pair of eye coordinates
        :return: first the left, then the right eye
        """
        first_eye, second_eye = eyes
        first_eye_left, second_eye_left = first_eye[0], second_eye[0]
        if first_eye_left < second_eye_left:
            return first_eye, second_eye
        else:
            return second_eye, first_eye

    @staticmethod
    def get_centered_box(image_width, image_height, box_width=0, box_height=0):
        """
        Returns a bounding box centered regarding the submitted image dimensions.

        :param image_width: the image width
        :param image_height: the image height
        :param box_width: the bounding box width
        :param box_height: the bounding box height
        :return: the bounding box coordinates consisting of left, top, right, bottom
        """
        horizontal_center = image_width // 2
        vertical_center = image_height // 2
        horizontal_distance_to_center = box_width // 2
        vertical_distance_to_center = box_height // 2

        box_left = horizontal_center - horizontal_distance_to_center
        box_top = vertical_center - vertical_distance_to_center
        box_right = horizontal_center + horizontal_distance_to_center
        box_bottom = vertical_center + vertical_distance_to_center

        return box_left, box_top, box_right, box_bottom
