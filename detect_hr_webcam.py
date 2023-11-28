# Must be run in console to work properly

import numpy as np
import cv2
import time
from scipy import signal
import threading

import scipy.signal as sig

WINDOW_NAME = 'Heartrate Monitoring'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def apply_ff(data, sample_frequency):
    
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


def get_heart_rate(bpm_list=list()):
    fs = 1 / (sum(camera_times) / data_length)
    temp_intensities = sig.detrend(apply_ff(intensities, fs))
    frequencies, pows = signal.welch(temp_intensities, fs=fs, nperseg=256)
    bpm = round(frequencies[np.argmax(pows)] * 60, 2)
    print("output BPM: ", bpm, fs)
    return bpm


def get_headbox_from_head(face):
    return (face[0] + face[2] // 4,
            face[1] + face[3] // 2,
            face[0] + 3 * face[2] // 4,
            face[1] + face[3] // 2 + 50)


video_capture = cv2.VideoCapture(0)


def read_intensity(intensities, current_frame, bounding_box):
    now = 0

    eye_left = 0
    head_top = 0
    eye_right = 0
    eye_top = 0
    while True:

        ret, frame = video_capture.read()

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

            scaling = .4
            eye_left = int(temp_headbox[0] * scaling + (1 - scaling) * eye_left)
            head_top = int(temp_headbox[1] * scaling + (1 - scaling) * head_top)
            eye_right = int(temp_headbox[2] * scaling + (1 - scaling) * eye_right)
            eye_top = int(temp_headbox[3] * scaling + (1 - scaling) * eye_top)

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
with open("data.txt", "w") as f:
    while True:
        frame = current_frame[0]
        bb = bounding_box[0]
        ROI = frame[bb[0]:bb[1], bb[2]:bb[3], 1]
        f.write(str(get_heart_rate()) + "\n")

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
