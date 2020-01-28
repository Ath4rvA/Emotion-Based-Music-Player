import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


# read model
read_model = open('my_model.json', 'r').read()

model = model_from_json(read_model)

# read weights
model.load_weights('weights.h5')

# face cascade
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#  live video
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read() # capture frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray image

    face_detected = face_haar_cascade.detectMultiScale(gray_frame)  # detect face from frame

    for (x,y,w,h) in face_detected:
        cv2.rectangle(frame, (x,y), (x+w, y+w), (0,255,0), 5)
        cropped_img = gray_frame[y:y+w, x:x+h]
        cropped_img = cv2.resize(cropped_img, (48,48))
        test_image = image.img_to_array(cropped_img)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255


        # predictions
        predictions = model.predict(test_image)
        # print(predictions)

        # max of predictions
        max_index = np.argmax(predictions[0])

        # lables
        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

        # predicted emotion
        predicted_emotion = emotions[max_index]

        # write lable on frame
        cv2.putText(frame, predicted_emotion, (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('test', resized_img)

    if cv2.waitKey(1) == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()


