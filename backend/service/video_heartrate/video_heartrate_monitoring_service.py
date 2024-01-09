import numpy as np

import cv2
import time
import threading

from scipy import signal

from backend.service.monitoring_service import MonitoringService
from backend.util.coordinate_utils import CoordinateUtils
from backend.util.moving_average import MovingAverage
from backend.service.util.result_type import ResultType

from backend.service.video_heartrate.eye_box_strategy import EyeBoxStrategy
from backend.parameter_settings import VIDEO_HEARTRATE_MONITORING_SERVICE_SETTINGS
from backend.style.style import Style
from backend.video.video_feed import VideoFeed


class VideoHeartrateMonitoringService(MonitoringService):
    """ Class that implements heartrate monitoring from a video feed.

    This class implements heartrate monitoring from a video feed. Face (and eye
    detection) are used to select a specific part of the face which is then used
    to calculate intensities. These intensities are then used to approximate the
    heartrate of the person associated with the detected face.
    """
    def __init__(
            self,
            video_feed: VideoFeed,
            name='VideoHeartrateMonitoring',
            settings=VIDEO_HEARTRATE_MONITORING_SERVICE_SETTINGS,
    ):
        """
        Create the video heartrate monitoring service.

        :param video_feed: the video feed
        :param name: the name of this service
        :param settings: the parameter settings of this service
        """
        super().__init__(
            name=name,
            result_types=[
                ResultType.HEARTRATE,
                ResultType.MOVING_AVERAGE_HEARTRATE,
                ResultType.FRAMERATE,
                ResultType.BOUNDING_BOX
            ],
            settings=settings
        )
        self.video_feed = video_feed

    def _initialize(self):
        """
        Initialize the video heartrate monitoring service.

        :return: True if the initialization was successful, False otherwise
        """
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        self.intensities_window_size = self.settings.get_value('IntensitiesWindowSize')
        self.camera_times = [0] * self.intensities_window_size
        self.intensities = []
        self.bounding_boxes = []
        self.moving_average_heartrate = MovingAverage(window_size=self.settings.get_value('MovingAverageHeartrateWindowSize'))

        self.is_running = False
        self.thread = threading.Thread(
            target=self._extract_intensities,
            daemon=True
        )

        return True

    def _run(self):
        """
        Start the video heartrate monitoring service.

        :return: True if starting the service was successful, False otherwise
        """
        self.is_running = True
        self.thread.start()

        return True

    def _halt(self):
        """
        Stop the video heartrate monitoring service.

        :return: True if stopping the service was successful, False otherwise
        """
        self.is_running = False
        self.thread.join()

        return True

    def _fetch(self):
        """
        Fetch the current results of this service.

        :return: a dictionary containing the results
        """
        bpm, fs = self._get_heart_rate()
        self.moving_average_heartrate.add(bpm)
        return {
            ResultType.HEARTRATE: bpm,
            ResultType.MOVING_AVERAGE_HEARTRATE: self.moving_average_heartrate.get(),
            ResultType.FRAMERATE: fs,
            ResultType.BOUNDING_BOX: self.boxes_to_display
        }

    def _extract_intensities(self):
        """
        A loop that extracts the intensity of the latest video feed frame and
        adds it to the list of intensities for heartrate calculation.
        """
        # image_scaling_factor = self.settings.get_value('ImageScalingFactor')
        bounding_box_update_rate = self.settings.get_value('BoundingBoxUpdateRate')

        frame_width = self.video_feed.get_frame_width()
        frame_height = self.video_feed.get_frame_height()

        face_box = CoordinateUtils.get_centered_box(frame_width, frame_height)
        left_eye_box = CoordinateUtils.get_centered_box(frame_width, frame_height)
        right_eye_box = CoordinateUtils.get_centered_box(frame_width, frame_height)
        roi_box = CoordinateUtils.get_centered_box(frame_width, frame_height)
        boxes_to_display = []

        now = 0
        while self.is_running:
            # fetch the next frame
            frame = self.video_feed.get_latest_frame()

            # scale down the image
            # frame = cv2.resize(frame, (-1, -1), fx=image_scaling_factor, fy=image_scaling_factor)

            face = self._get_face_coordinates(frame)
            if face is not None:
                face_box = CoordinateUtils.update_coordinates(face_box, face, bounding_box_update_rate)
                if self.settings.get_value('DisplayFaceBox'):
                    boxes_to_display.append((face_box, Style.FACE_BOX_STYLE))

                eyes = self._get_eye_coordinates(frame, face)
                if eyes is not None:
                    left_eye, right_eye = eyes
                    left_eye_box = CoordinateUtils.update_coordinates(left_eye_box, left_eye, bounding_box_update_rate)
                    right_eye_box = CoordinateUtils.update_coordinates(right_eye_box, right_eye, bounding_box_update_rate)
                    if self.settings.get_value('DisplayLeftEyeBox'):
                        boxes_to_display.append((left_eye_box, Style.EYE_BOX_STYLE))
                    if self.settings.get_value('DisplayRightEyeBox'):
                        boxes_to_display.append((right_eye_box, Style.EYE_BOX_STYLE))

                roi_box = CoordinateUtils.update_coordinates(roi_box, self._get_roi_box(face, eyes), bounding_box_update_rate)
                if self.settings.get_value('DisplayRoiBox'):
                    boxes_to_display.append((roi_box, Style.ROI_BOX_STYLE))

                # extract the region of interest
                roi_box_left, roi_box_top, roi_box_right, roi_box_bottom = roi_box
                roi = frame[roi_box_top:roi_box_bottom, roi_box_left:roi_box_right, 1]
                intensity = roi.mean()  # intensity = np.median(roi) works, but quite chunky
                self.intensities.append(intensity)

                if len(self.intensities) > self.intensities_window_size:
                    self.intensities.pop(0)

                self.camera_times.append(time.time() - now)
                now = time.time()
                self.camera_times.pop(0)

            self.boxes_to_display = boxes_to_display
            boxes_to_display = []

    def _get_face_coordinates(self, image):
        """
        Detects a face in an image and returns its coordinates.

        :param image: the image to be examined
        :return: the coordinates of the first detected face as a tuple consisting of left, top, right, bottom
        """
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Only take one face, the first
        faces = self.face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
        if len(faces) < 1:
            return None

        return CoordinateUtils.convert_coordinates(faces[0])

    def _get_eye_coordinates(self, image, face):
        """
        Searches for eyes in the area of an image where a face was detected. If at
        least two are found, the first two that got detected are returned.

        :param image: the image to be examined
        :param face: the face coordinates
        :return: the coordinates of the first detected pair of eyes as tuples consisting of left, top, right, bottom
        """
        face_left, face_top, face_right, face_bottom = face

        # convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect eyes in the cropped out face
        cropped_grayscale_image = grayscale_image[face_top:face_bottom, face_left:face_right]
        eyes, number_of_detections = self.eye_cascade.detectMultiScale2(cropped_grayscale_image, minNeighbors=10)

        if len(number_of_detections) < 2:
            return None

        # put the eyes in the correct order
        left_eye, right_eye = CoordinateUtils.order_eyes((eyes[0], eyes[1]))

        # adjust coordinates to be at the correct place in the original image
        left_eye[0] += face_left
        left_eye[1] += face_top
        right_eye[1] += face_top
        right_eye[0] += face_left

        return CoordinateUtils.convert_coordinates(left_eye), CoordinateUtils.convert_coordinates(right_eye)

    def _get_roi_box(self, face, eyes):
        """
        Extract the region of interest from face and eye coordinates.

        :param face: the face coordinates
        :param eyes: the eye coordinates
        :return: the coordinates of the region of interest as a tuple consisting of left, top, right and bottom
        """
        face_left, face_top, face_right, face_bottom = face
        face_width = face_right - face_left
        face_height = face_bottom - face_top

        # (fallback) approximation strategy
        box_left = face_left + (face_width // 4)
        box_top = face_top + (face_height // 3)
        box_right = face_left + (3 * face_width // 4)
        box_bottom = face_top + (face_height // 2 + round(face_height * 0.15))

        # (fallback) original approximation strategy
        # box_left = face_left + (face_width // 4)
        # box_top = face_top + (face_height // 3)
        # box_right = face_left + (3 * face_width // 4)
        # box_bottom = face_top + (face_height // 2 + 50)

        if eyes is not None:

            left_eye, right_eye = eyes
            left_eye_left, left_eye_top, _, left_eye_bottom = left_eye
            _, right_eye_top, right_eye_right, right_eye_bottom = right_eye

            if self.settings.get_value('EyeBoxStrategy') == EyeBoxStrategy.STRICT:

                box_left = left_eye_left
                box_top = max(left_eye_top, right_eye_top)
                box_right = right_eye_right

            elif self.settings.get_value('EyeBoxStrategy') == EyeBoxStrategy.CONTRACTION:

                eye_box_width = right_eye_right - left_eye_left
                eye_box_height = max(left_eye_bottom, right_eye_bottom) - min(left_eye_top, right_eye_top)

                horizontal_contraction_factor = self.settings.get_value('EyeBoxHorizontalContractionFactor')
                vertical_contraction_factor = self.settings.get_value('EyeBoxVerticalContractionFactor')

                box_left = left_eye_left + int(eye_box_width * horizontal_contraction_factor / 2)
                box_top = min(max(left_eye_top, right_eye_top) + int(eye_box_height * vertical_contraction_factor), box_bottom)
                box_right = right_eye_right - int(eye_box_width * horizontal_contraction_factor / 2)

            elif self.settings.get_value('EyeBoxStrategy') == EyeBoxStrategy.APPROXIMATION:

                approximation_ratio = self.settings.get_value('EyeBoxApproximationRatio')

                box_left = int(CoordinateUtils.get_weighted_average(box_left, left_eye_left, approximation_ratio))
                box_top = int(CoordinateUtils.get_weighted_average(box_top, max(left_eye_top, right_eye_top), approximation_ratio))
                box_right = int(CoordinateUtils.get_weighted_average(box_right, right_eye_right, approximation_ratio))

        return box_left, box_top, box_right, box_bottom

    def _apply_ff(self, data, sample_frequency):
        """
        Applies a forward-backward digital filter using cascaded second-order sections.

        :param data: the data
        :param sample_frequency: the sample frequency
        :return: the filtered data
        """
        if sample_frequency > 3:
            sos = signal.iirdesign(
                [.66, 3.0],
                [.5, 4.0],
                1.0,
                40.0,
                fs=sample_frequency,
                output='sos'
            )
            return signal.sosfiltfilt(sos, data)
        else:
            return data

    def _get_heart_rate(self):
        """
        Calculates the heart rate using intensities extracted from face images.

        :return: heart rates in bpm
        """
        fs = self.intensities_window_size / sum(self.camera_times)
        temp_intensities = signal.detrend(self._apply_ff(np.nan_to_num(self.intensities), fs))
        frequencies, pows = signal.welch(temp_intensities, fs=fs, nperseg=256)
        bpm = round(frequencies[np.argmax(pows)] * 60, 2)
        return bpm, fs
