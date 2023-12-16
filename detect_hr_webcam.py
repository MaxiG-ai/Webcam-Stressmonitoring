# Must be run in console to work properly

from enum import Enum

import numpy as np
import cv2
import time
from scipy import signal
import threading

import scipy.signal as sig

import os
from tkinter import filedialog


class VideoSource(Enum):
    WEBCAM = 1
    FILE = 2


WINDOW_NAME = 'Heartrate Monitoring'
VIDEO_SOURCE = VideoSource.WEBCAM
DISPLAY_FACE_BOX = True
DISPLAY_ROI_BOX = True
# TIMEOUT = 3600  # 1 hour
# TIMEOUT = 60 # 60 seconds
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
FACE_BOX_STYLE = {'color': GREEN, 'thickness': 2}
ROI_BOX_STYLE = {'color': GREEN, 'thickness': 1}

data_length = 120
camera_times = [0] * data_length
intensities = []
x = list(range(len(intensities)))

bpm_history_rolling = np.zeros(120)

if VIDEO_SOURCE == VideoSource.WEBCAM:
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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


def get_face_and_eye_coordinates(image):
    """
    Detects a face and a pair of eyes in an image and returns their coordinates.

    :param image: the image to be examined
    :return: the coordinates of the first detected face and pair of eyes
    """
    # convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Only take one face, the first
    faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
    if len(faces) < 1:
        return None
    (x, y, w, h) = faces[0]

    # Crop out faces and detect eyes in that window.
    roi_grayscale_image = grayscale_image[y:y + h, x:x + w]
    eyes, number_of_detections = eye_cascade.detectMultiScale2(roi_grayscale_image, minNeighbors=10)
    if len(number_of_detections) < 2:
        return None

    # Change eye coords to be in image coordinates instead of face coordinates
    eyes[0][0] += x
    eyes[1][0] += x
    eyes[0][1] += y
    eyes[1][1] += y

    return [faces[0], eyes[0], eyes[1]]


def get_face_coordinates(image):
    """
    Detects a face in an image and returns its coordinates.

    :param image: the image to be examined
    :return: the coordinates of the first detected face as a tuple consisting of left, top, width and height
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Only take one face, the first
    faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
    if len(faces) < 1:
        return None

    return faces[0]


def get_heart_rate():
    """
    Calculates the heart rate using intensities extracted from face images.

    :return: heart rates in bpm
    """
    fs = 1 / (sum(camera_times) / data_length)
    temp_intensities = sig.detrend(apply_ff(intensities, fs))
    frequencies, pows = signal.welch(temp_intensities, fs=fs, nperseg=256)
    bpm = round(frequencies[np.argmax(pows)] * 60, 2)
    global bpm_history_rolling
    bpm_history_rolling = np.append(bpm_history_rolling, bpm)
    rolling_avg_bpm = np.mean(bpm_history_rolling[-120:])
    print("output BPM: ", bpm, rolling_avg_bpm, fs)
    return rolling_avg_bpm


def get_roi_box_from_face(face):
    """
    Extract the region of interest from face coordinates.

    :param face: the face coordinates
    :return: the coordinates of the region of interest as a tuple consisting of left, top, right and bottom
    """
    face_left = face[0]
    face_top = face[1]
    face_width = face[2]
    face_height = face[3]

    # box_left = face_left
    # box_top = face_top
    # box_right = face_left + face_width
    # box_bottom = face_top + face_height

    box_left = face_left + (face_width // 4)
    box_top = face_top + (face_height // 3)
    box_right = face_left + (3 * face_width // 4)
    box_bottom = face_top + (face_height // 2 + round(face_height * 0.15))

    # box_left = face_left + (face_width // 4)
    # box_top = face_top + (face_height // 3)
    # box_right = face_left + (3 * face_width // 4)
    # box_bottom = face_top + (face_height // 2 + 50)

    return box_left, box_top, box_right, box_bottom


def convert_face_coordinates(face):
    """
    Converts the face coordinates.

    :param face: the face coordinates as a tuple consisting of left, top, width, height
    :return: the face coordinates as a tuple consisting of left, top, right, bottom
    """
    left, top, width, height = face
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

    updated_left = int(new_left * update_rate + (1 - update_rate) * old_left)
    updated_top = int(new_top * update_rate + (1 - update_rate) * old_top)
    updated_right = int(new_right * update_rate + (1 - update_rate) * old_right)
    updated_bottom = int(new_bottom * update_rate + (1 - update_rate) * old_bottom)

    return updated_left, updated_top, updated_right, updated_bottom


def read_intensity(intensities, current_frame, bounding_box):
    """
    Extracts intensities from the face for calculating the heart rate.

    :param intensities: the intensities
    :param current_frame: the current frame
    :param bounding_box: the bounding box of the face
    """
    scale_factor = 0.4  # determines how much the original image is scaled down
    update_rate = 0.4  # determines how fast bounding boxes adapt

    if VIDEO_SOURCE == VideoSource.FILE:
        frame_counter = 0

    face_box = (0, 0, 0, 0)
    roi_box = (0, 0, 0, 0)
    boxes_to_display = []

    now = 0
    while True:
        # fetch the next frame
        _, frame = video_capture.read()

        # if a video file is used as source, it is restarted when the last frame is reached
        if VIDEO_SOURCE == VideoSource.FILE:
            frame_counter += 1
            if frame_counter == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # scale down the image
        frame = cv2.resize(frame, (-1, -1), fx=scale_factor, fy=scale_factor)

        face = get_face_coordinates(frame)
        if face is not None:
            # update the box coordinates with the new ones
            face_box = update_coordinates(face_box, convert_face_coordinates(face), update_rate)
            roi_box = update_coordinates(roi_box, get_roi_box_from_face(face), update_rate)

            # extract the region of interest
            roi_box_left, roi_box_top, roi_box_right, roi_box_bottom = roi_box
            roi = frame[roi_box_top:roi_box_bottom, roi_box_left:roi_box_right, 1]
            intensity = roi.mean()  # intensity = np.median(roi) works, but quite chunky
            intensities.append(intensity)

            if DISPLAY_FACE_BOX:
                boxes_to_display.append((face_box, FACE_BOX_STYLE))
            if DISPLAY_ROI_BOX:
                boxes_to_display.append((roi_box, ROI_BOX_STYLE))

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
            f.write(str(get_heart_rate()) + "\n")

            cv2.imshow(WINDOW_NAME, frame)
            # if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > TIMEOUT:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
