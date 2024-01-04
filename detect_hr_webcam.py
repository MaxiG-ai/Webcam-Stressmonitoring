# Must be run in console to work properly

import warnings
warnings.filterwarnings("ignore")

from enum import Enum

import numpy as np
import cv2
import time
from scipy import signal
import threading

import scipy.signal as sig

import os
from tkinter import filedialog


class MovingAverage:
    """
    A class for calculating the moving average of a data stream.
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
        Add a value to the moving average.

        :param value: the value
        """
        self.history = np.append(self.history, value)

    def get(self):
        """
        Get the current moving average.

        :return: the moving average
        """
        return np.mean(self.history[-self.window_size:])


class VideoSource(Enum):
    WEBCAM = 1
    FILE = 2
    DEMO = 3


VIDEO_SOURCE = VideoSource.DEMO
DEMO_FILE_PATH = os.path.join('data', 'heartrate', 'face_1.mp4')
LOOP_VIDEO_IF_POSSIBLE = True
DISPLAY_FACE_BOX = True
DISPLAY_ROI_BOX = True
DISPLAY_LEFT_EYE_BOX = True
DISPLAY_RIGHT_EYE_BOX = True


class EyeBoxStrategy(Enum):
    # left, top and right of eye box are the left, top and right bounds of the roi box
    STRICT = 1
    # eye box is shrunk by fixed factors and used as the left, top and right bounds of the roi box
    CONTRACTION = 2
    # left, top and right are determined by the weighted average eye box and approximation
    APPROXIMATION = 3
    # only use the approximation
    DISABLED = 4


# the strategy that determines how the eye box is used for determining the roi for video_heartrate detection
EYE_BOX_STRATEGY = EyeBoxStrategy.CONTRACTION
# determines by how much the eye box is shrunk horizontally when using EyeBoxStrategy.CONTRACTION
EYE_BOX_HORIZONTAL_CONTRACTION_FACTOR = 0.1
# determines by how much the eye box is shrunk vertically when using EyeBoxStrategy.CONTRACTION
EYE_BOX_VERTICAL_CONTRACTION_FACTOR = 0.25
# used for calculating the weighted average when using EyeBoxStrategy.APPROXIMATION
EYE_BOX_APPROXIMATION_RATIO = 0.4

# TIMEOUT = 3600  # 1 hour
# TIMEOUT = 60 # 60 seconds
WINDOW_NAME = 'Heartrate Monitoring'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

GREEN = (0, 255, 0)
YELLOW = (0, 255, 192)
FACE_BOX_STYLE = {'color': GREEN, 'thickness': 2}
EYE_BOX_STYLE = {'color': GREEN, 'thickness': 1}
ROI_BOX_STYLE = {'color': YELLOW, 'thickness': 1}

INITIAL_BPM = 0
MOVING_AVERAGE_WINDOW = 512
moving_average_bpm = MovingAverage(INITIAL_BPM, MOVING_AVERAGE_WINDOW)

data_length = 120
camera_times = [0] * data_length
intensities = []
x = list(range(len(intensities)))

if VIDEO_SOURCE == VideoSource.WEBCAM:
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    looping_video_possible = False
elif VIDEO_SOURCE == VideoSource.FILE:
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        filetypes=[
            ('Video files', ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'))
        ]
    )
    if file_path == '' or file_path is None:
        print('No file was selected. Stopping execution.')
        exit(1)
    video_capture = cv2.VideoCapture(file_path)
    looping_video_possible = True
elif VIDEO_SOURCE == VideoSource.DEMO:
    file_path = os.path.join(os.getcwd(), DEMO_FILE_PATH)
    if not os.path.exists(file_path):
        print('The file does not exist. Stopping execution.')
        exit(1)
    video_capture = cv2.VideoCapture(file_path)
    looping_video_possible = True
else:
    print('No video source was defined. Stopping execution.')
    exit(1)


def apply_ff(data, sample_frequency):
    """
    Applies a forward-backward digital filter using cascaded second-order sections.

    :param data: the data
    :param sample_frequency: the sample frequency
    :return: the filtered data
    """
    if sample_frequency > 3:
        sos = sig.iirdesign(
            [.66, 3.0],
            [.5, 4.0],
            1.0,
            40.0,
            fs=sample_frequency,
            output='sos'
        )
        return sig.sosfiltfilt(sos, data)
    else:
        return data


def get_face_coordinates(image):
    """
    Detects a face in an image and returns its coordinates.

    :param image: the image to be examined
    :return: the coordinates of the first detected face as a tuple consisting of left, top, right, bottom
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Only take one face, the first
    faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
    if len(faces) < 1:
        return None

    return convert_coordinates(faces[0])


def get_eye_coordinates(image, face):
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
    from PIL import Image
    eyes, number_of_detections = eye_cascade.detectMultiScale2(cropped_grayscale_image, minNeighbors=10)

    if len(number_of_detections) < 2:
        return None

    # put the eyes in the correct order
    left_eye, right_eye = order_eyes((eyes[0], eyes[1]))

    # adjust coordinates to be at the correct place in the original image
    left_eye[0] += face_left
    left_eye[1] += face_top
    right_eye[1] += face_top
    right_eye[0] += face_left

    return convert_coordinates(left_eye), convert_coordinates(right_eye)


def get_heart_rate():
    """
    Calculates the heart rate using intensities extracted from face images.

    :return: heart rates in bpm
    """
    fs = 1 / (sum(camera_times) / data_length)
    temp_intensities = sig.detrend(apply_ff(intensities, fs))
    frequencies, pows = signal.welch(temp_intensities, fs=fs, nperseg=256)
    bpm = round(frequencies[np.argmax(pows)] * 60, 2)
    return bpm, fs


def get_roi_box(face, eyes):
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

        if EYE_BOX_STRATEGY == EyeBoxStrategy.STRICT:

            box_left = left_eye_left
            box_top = max(left_eye_top, right_eye_top)
            box_right = right_eye_right

        elif EYE_BOX_STRATEGY == EyeBoxStrategy.CONTRACTION:

            eye_box_width = right_eye_right - left_eye_left
            eye_box_height = max(left_eye_bottom, right_eye_bottom) - min(left_eye_top, right_eye_top)

            box_left = left_eye_left + int(eye_box_width * EYE_BOX_HORIZONTAL_CONTRACTION_FACTOR / 2)
            box_top = min(max(left_eye_top, right_eye_top) + int(eye_box_height * EYE_BOX_VERTICAL_CONTRACTION_FACTOR), box_bottom)
            box_right = right_eye_right - int(eye_box_width * EYE_BOX_HORIZONTAL_CONTRACTION_FACTOR / 2)

        elif EYE_BOX_STRATEGY == EyeBoxStrategy.APPROXIMATION:

            box_left = int(get_weighted_average(box_left, left_eye_left, EYE_BOX_APPROXIMATION_RATIO))
            box_top = int(get_weighted_average(box_top, max(left_eye_top, right_eye_top), EYE_BOX_APPROXIMATION_RATIO))
            box_right = int(get_weighted_average(box_right, right_eye_right, EYE_BOX_APPROXIMATION_RATIO))

    return box_left, box_top, box_right, box_bottom


def convert_coordinates(coordinates):
    """
    Converts coordinates of the form (left, top, width, height) to coordinates
    of the form (left, top, right, bottom).

    :param coordinates: the coordinates as a tuple consisting of left, top, width, height
    :return: the coordinates as a tuple consisting of left, top, right, bottom
    """
    left, top, width, height = coordinates
    return left, top, left + width, top + height


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

    updated_left = int(get_weighted_average(new_left, old_left, update_rate))
    updated_top = int(get_weighted_average(new_top, old_top, update_rate))
    updated_right = int(get_weighted_average(new_right, old_right, update_rate))
    updated_bottom = int(get_weighted_average(new_bottom, old_bottom, update_rate))

    return updated_left, updated_top, updated_right, updated_bottom


def get_weighted_average(x, y, ratio):
    """
    Calculates the weighted average between two values according to a given ratio.

    :param x: the first value
    :param y: the second value
    :param ratio: the ratio
    :return: the weighted average
    """
    return x * ratio + y * (1 - ratio)


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


def read_intensity(intensities, current_frame, bounding_box):
    """
    Extracts intensities from the face for calculating the heart rate.

    :param intensities: the intensities
    :param current_frame: the current frame
    :param bounding_box: the bounding box of the face
    """
    scale_factor = 0.4  # determines how much the original image is scaled down
    update_rate = 0.4  # determines how fast bounding boxes adapt

    if LOOP_VIDEO_IF_POSSIBLE and looping_video_possible:
        frame_counter = 0

    frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    face_box = get_centered_box(frame_width, frame_height)
    left_eye_box = get_centered_box(frame_width, frame_height)
    right_eye_box = get_centered_box(frame_width, frame_height)
    roi_box = get_centered_box(frame_width, frame_height)
    boxes_to_display = []

    now = 0
    while True:
        # fetch the next frame
        _, frame = video_capture.read()

        # if a video file is used as source, it is restarted when the last frame is reached
        if LOOP_VIDEO_IF_POSSIBLE and looping_video_possible:
            frame_counter += 1
            if frame_counter == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # scale down the image
        frame = cv2.resize(frame, (-1, -1), fx=scale_factor, fy=scale_factor)

        face = get_face_coordinates(frame)
        if face is not None:
            face_box = update_coordinates(face_box, face, update_rate)
            if DISPLAY_FACE_BOX:
                boxes_to_display.append((face_box, FACE_BOX_STYLE))

            eyes = get_eye_coordinates(frame, face)
            if eyes is not None:
                left_eye, right_eye = eyes
                left_eye_box = update_coordinates(left_eye_box, left_eye, update_rate)
                right_eye_box = update_coordinates(right_eye_box, right_eye, update_rate)
                if DISPLAY_LEFT_EYE_BOX:
                    boxes_to_display.append((left_eye_box, EYE_BOX_STYLE))
                if DISPLAY_RIGHT_EYE_BOX:
                    boxes_to_display.append((right_eye_box, EYE_BOX_STYLE))

            roi_box = update_coordinates(roi_box, get_roi_box(face, eyes), update_rate)
            if DISPLAY_ROI_BOX:
                boxes_to_display.append((roi_box, ROI_BOX_STYLE))

            # extract the region of interest
            roi_box_left, roi_box_top, roi_box_right, roi_box_bottom = roi_box
            roi = frame[roi_box_top:roi_box_bottom, roi_box_left:roi_box_right, 1]
            intensity = roi.mean()  # intensity = np.median(roi) works, but quite chunky
            intensities.append(intensity)

            # expand bounding box slightly
            bounding_box[0] = [
                roi_box_top + 2,
                roi_box_bottom - 2,
                roi_box_left + 2,
                roi_box_right - 2
            ]

            if len(intensities) > data_length:
                intensities.pop(0)

            camera_times.append(time.time() - now)
            now = time.time()
            camera_times.pop(0)

        # add the detected bounding boxes to the frame
        for ((box_left, box_top, box_right, box_bottom), style) in boxes_to_display:
            frame = cv2.rectangle(
                frame,
                (box_left, box_top),
                (box_right, box_bottom),
                **style
            )
        boxes_to_display = []

        # display the frame
        current_frame[0] = frame


if __name__ == "__main__":
    bounding_box = [0]
    current_frame = [0]
    thread = threading.Thread(
        target=read_intensity,
        daemon=True,
        args=(intensities, current_frame, bounding_box)
    )
    thread.start()

    time.sleep(1)
    # start_time = time.time()
    print("Camera fs: ", 1 / (sum(camera_times) / data_length))
    with open("data.txt", "w") as f:
        while True:
            frame = current_frame[0]
            bb = bounding_box[0]
            ROI = frame[bb[0]:bb[1], bb[2]:bb[3], 1]
            bpm, fs = get_heart_rate()
            moving_average_bpm.add(bpm)

            output = "Moving Average BPM: {:.2f}, BPM: {:.2f}, FS: {:.2f}".format(moving_average_bpm.get(), bpm, fs)
            print(output)
            f.write(output + "\n")

            cv2.imshow(WINDOW_NAME, frame)
            # if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > TIMEOUT:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
