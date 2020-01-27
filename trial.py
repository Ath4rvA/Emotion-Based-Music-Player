#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:01:15 2019

@author: atharva
"""

import os
import random
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3


# read model
read_model = open('my_model.json', 'r').read()

model = model_from_json(read_model)

# read weights
model.load_weights('weights.h5')

# face cascade
face_haar_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

#  live video
# capture = cv2.VideoCapture(0)


def prediction(img):

    # engine= pyttsx3.init()
    # voices = engine.getProperty('voices')
    # voices[11].gender='female'
    # engine.setProperty('voice', voices[11].id)
    # engine.setProperty('gender', 1)
    # rate = engine.getProperty('rate')
    # engine.setProperty('rate', 180)

    #     ret, frame = capture.read() # capture frame
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray image

    face_detected = face_haar_cascade.detectMultiScale(
        gray_frame)  # detect face from frame
   # print(type(face_detected))
    test_image = None
    for (x, y, w, h) in face_detected:
        cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 5)
        cropped_img = gray_frame[y:y + w, x:x + h]
        cropped_img = cv2.resize(cropped_img, (48, 48))
        test_image = image.img_to_array(cropped_img)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255

       # predictions
    # print(test_image)
    predictions = model.predict(test_image, steps=1)
    print(predictions)
    # return "HAHA"
    # # max of predictions
    max_index = np.argmax(predictions[0])
    # print(max_index)
    # # lables
    emotions = ('angry', 'disgust', 'fear', 'happy',
                'neutral', 'sad', 'surprised')

    # # predicted emotion
    predicted_emotion = emotions[max_index]
    print(predicted_emotion)

    # # write lable on frame
    # #cv2.putText(img, predicted_emotion, (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    base_add = '/home/atharva/Documents/Project/GUI/static/music'
    genre_url = base_add + '/' + predicted_emotion
    songs_list = os.listdir(genre_url)
    # # print(songs_list)
    choice = random.choice(songs_list)
    test = genre_url + '/' + choice
    results = test.split('static')
    final = 'static' + results[1]
    return final, predicted_emotion
    # play(sound)
    # resized_img = cv2.resize(img, (1000, 700))
    # cv2.imshow('test', resized_img)

    # if cv2.waitKey(1) == ord('q'):
    #   break

   # capture.release()
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    # engine.say(f"Ohhh..... your mood seems to be {predicted_emotion}")
    # engine.say("...Let me play a song for you")
    # engine.runAndWait()
    # play(sound)


# def capture_picture():
#     cam = cv2.VideoCapture(0)

#     cv2.namedWindow("test")

#     #img_counter = 0

#     while True:
#         ret, frame = cam.read()
#         cv2.imshow("test", frame)
#         if not ret:
#             break
#         k = cv2.waitKey(1)

#         if k % 256 == 27:
#             # ESC pressed
#             print("Escape hit, closing...")
#             break
#         elif k % 256 == 32:
#             img = frame
#             cam.release()
#             cv2.destroyAllWindows()
#             return img

#     cam.release()
#     cv2.destroyAllWindows()


# img = capture_picture()
# img = cv2.imread('/tmp/image.jpg')
# prediction(img)
# =============================================================================
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================
