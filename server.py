from flask import Flask, render_template, request
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import cv2
from keras.models import model_from_json
from keras.preprocessing import image
import random
import numpy as np
from trial import prediction
import os

import pandas as pd
import mysql_backend_connector
app = Flask(__name__, static_url_path="/static")

connector = mysql_backend_connector.MysqlBackendConnector(
    "emoplayer", "dbda", "dbda")


read_model = open('my_model.json', 'r').read()

model = model_from_json(read_model)

# read weights
model.load_weights('weights.h5')
face_haar_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


@app.route("/login", methods=["POST"])
def login():
    if connector.login(request.form['username'], request.form['password']):

        return render_template('client.html')
    else:
        return "<h1>Error! Please check your credentials.</h1>"


@app.route('/searchQuery', methods=['POST'])
def search():
    file = request.form['file']
    starter = file.find(',')
    image_data = file[starter + 1:]
    image_data = bytes(image_data, encoding="ascii")
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im.save('static/image.jpg')
    return '<h1>uploaded</h1>'


@app.route('/play', methods=['POST'])
def play():
    img = cv2.imread('static/image.jpg')
    path = None
    emotion = None
    try:
        path, emotion = prediction(img)
        print(path, emotion)
        return render_template('music.html', mssrc=path, emotion=emotion)
    except:
        print(path, emotion)
        return '<h1>Please go back and re-try capturing</h1>'


@app.route("/signup", methods=["GET", "POST"])
def signup():
    data = request.form
    if data['password'] == data['cpassword']:
        connector.signup(data['email'], data['password'])
        return render_template("login.html")
    else:
        return "<h1>Error! Please check your confirmed password!</h1>"


@app.route("/signup_page")
def signup_page():
    return render_template("signup.html")


@app.route("/login_page")
def login_page():
    return render_template("login.html")


@app.route('/')
def index():
    return render_template('index_.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/FAQ')
def FAQ():
    return render_template('FAQ.html')


@app.route('/team')
def team():
    return render_template('team.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)
