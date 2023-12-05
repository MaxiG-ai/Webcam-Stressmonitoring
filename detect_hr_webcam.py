# Must be run in console to work properly

import numpy as np
import cv2
import time
from scipy import signal
import threading

import scipy.signal as sig

WINDOW_NAME = 'Heartrate Monitoring'
TIMEOUT = 60
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

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
    :return: the coordinates of the first detected face
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Only take one face, the first
    faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
    if len(faces) < 1:
        return None

    return faces[0]


data_length = 120
camera_times = [0] * data_length
intensities = []
x = list(range(len(intensities)))

bpm_history_rolling = np.zeros(120)

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


def get_headbox_from_head(face):
    """
    Extract the headbox from face coordinates.

    :param face: the face coordinates
    :return: the headbox
    """
    return (face[0] + face[2] // 4,
            face[1] + face[3] // 3,
            face[0] + 3 * face[2] // 4,
            face[1] + face[3] // 2 + 50)


video_capture = cv2.VideoCapture(0)


def read_intensity(intensities, current_frame, bounding_box):
    """
    Extracts intensities from the face for calculating the heart rate.

    :param intensities: the intensities
    :param current_frame: the current frame
    :param bounding_box: the bounding box of the face
    """
    now = 0

    eye_left = 0
    head_top = 0
    eye_right = 0
    eye_top = 0
    while True:
        # fetch the next frame
        _, frame = video_capture.read()

        scale_factor = 0.4
        frame = cv2.resize(frame, (-1, -1), fx=scale_factor, fy=scale_factor)

        # tmp = get_face_coordinates(frame) # Haar outputs [x, y, w, h] format
        face = get_face_coordinates(frame)
        if face is not None:
            # if tmp != None:
            # face, eye1, eye2 = tmp
            # eyeleft, headTop, eyeright, eyeTop\
            # temp_headbox = get_headbox_from_head(face, eye1, eye2)
            temp_headbox = get_headbox_from_head(face)

            update_rate = .4
            eye_left = int(temp_headbox[0] * update_rate + (1 - update_rate) * eye_left)
            head_top = int(temp_headbox[1] * update_rate + (1 - update_rate) * head_top)
            eye_right = int(temp_headbox[2] * update_rate + (1 - update_rate) * eye_right)
            eye_top = int(temp_headbox[3] * update_rate + (1 - update_rate) * eye_top)

            roi = frame[head_top:eye_top, eye_left:eye_right, 1]
            intensity = roi.mean()
            # intensity = np.median(roi) # works, but quite chunky.

            intensities.append(intensity)

            # Draw the forehead box:
            current_frame[0] = cv2.rectangle(
                frame,
                (eye_left, head_top),
                (eye_right, eye_top),
                (0, 255, 0),
                1
            )
            bounding_box[0] = [
                head_top + 2,
                eye_top - 2,
                eye_left + 2,
                eye_right - 2
            ]

            if (len(intensities) > data_length):
                intensities.pop(0)

            camera_times.append(time.time() - now)
            now = time.time()
            camera_times.pop(0)


bounding_box = [0]
current_frame = [0]
thread = threading.Thread(
    target=read_intensity,
    daemon=True,
    args=(intensities, current_frame, bounding_box)
)
thread.start()

time.sleep(1)
start_time = time.time()
print("Camera fs: ", 1 / (sum(camera_times) / data_length))
with open("data.txt", "w") as f:
    while True:
        frame = current_frame[0]
        bb = bounding_box[0]
        ROI = frame[bb[0]:bb[1], bb[2]:bb[3], 1]
        f.write(str(get_heart_rate()) + "\n")

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > TIMEOUT:
            break
