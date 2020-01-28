import numpy as np
import pandas as pd
import os
import cv2
from pathlib import Path
import split_folders

# read data from csv
df = pd.read_csv("/home/atharva/Documents/Project/Images/fer2013.csv")

# labels of emotion
labels = np.array(df['emotion'].values)

# convert labels from [0 1 2 3 4 5 6] to ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
labels = np.where(labels == 0, 'Angry', np.where(labels == 1, 'Disgust', np.where(labels == 2, 'Fear',
                  np.where(labels == 3, 'Happy', np.where(labels == 4, 'Sad', np.where(labels == 5, "Surprise", "Neutral"))))))

# print(labels)

# image data
pixels = list(df['pixels'].values)
test=[]  # list of images

# create image from pixel
for pixel in pixels:
    images = np.array([np.fromstring(pixel, dtype='uint8',sep=' ')])
    # print(images.shape)
    images.shape = (1, 48, 48)
    test.extend(images)
    # print(images.shape)
    # img = images[0]
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# zip image and labels  together
data = zip(labels, test)

# create directory to save images
if not os.path.exists('./images'):
    os.mkdir('./images')

# save images as per emotion
cnt = 0
for d in data:
    #print(d)
    dir = Path(f'./images/{d[0]}')
    if not os.path.exists(dir):
        os.mkdir(dir)
    cv2.imwrite(f'{dir}/img{cnt}.jpg', d[1])
    # img = cv2.imread(f'{dir}/img{cnt}.jpg')
    # img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(f'{dir}/gryimg{cnt}.jpg', img_gry)
    # print(img_gry.shape)
    cnt += 1

# split dataset into test train and validation
split_folders.ratio('./images/', output='./images/DATA', seed=12345, ratio=(0.8,0.1,0.1))
