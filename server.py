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

#from fetch_try import get_search_results
#from algo import train, predict, load_data
import pandas as pd
import mysql_backend_connector
app = Flask(__name__, static_url_path="/static")
#yt_df = ""
# nltk.download()

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
        # return render_template("index.html")
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
    print(img)
    # gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face_detected = face_haar_cascade.detectMultiScale(gray_frame)
    # for (x, y, w, h) in face_detected:
    #     cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0), 5)
    #     cropped_img = gray_frame[y:y + w, x:x + h]
    #     cropped_img = cv2.resize(cropped_img, (48, 48))
    #     test_image = image.img_to_array(cropped_img)
    #     test_image = np.expand_dims(test_image, axis=0)
    #     test_image /= 255

    # predictions = model.predict(test_image)
    # max_index = np.argmax(predictions[0])
    # emotions = ('angry', 'disgust', 'fear', 'happy',
    #             'neutral', 'sad', 'surprised')
    # predicted_emotion = emotions[max_index]
    # base_add = '/home/atharva/Documents/Project/GUI/static/music'
    # genre_url = base_add + '/' + predicted_emotion
    # songs_list = os.listdir(genre_url)
    # choice = random.choice(songs_list)
    try:
        path, emotion = prediction(img)
        return render_template('music.html', mssrc=path, emotion=emotion)
    except:
        return '<h1>Please go back and re-try capturing</h1>'
    # return render_template('music.html', mssrc=path)


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


'''
@app.route("/searchQuery",methods=['POST'])
def search():
    data=request.form
    print("Search text:",data['search_text'])
    df=get_search_results(data['search_text'])
    df=df.fillna(0)
    global yt_df
    yt_df=df
    x_tr,x_te,y_tr,y_te=load_data()
    train(x_tr,x_te,y_tr,y_te)
    df=predict(df)
    print(df)
#    import matplotlib.pyplot as plt
#    import pandas as pd
#    from pandas.plotting import table
#    ax = plt.subplot(111, frame_on=False) # no visible frame
#    ax.xaxis.set_visible(False)  # hide the x axis
#    ax.yaxis.set_visible(False)  # hide the y axis
#    table(ax, df)  # where df is your data frame
#    plt.savefig('mytable.png')
#    return render_template('show.html')
    #new_df= df.sort_values(by=['score'], ascending= False)
    new_df=df
    new_df.set_index('score',inplace=True)
    new_df=new_df.sort_values(by=['score'],ascending=False)
    for i in range(len(new_df)):
        temp=new_df.iloc[i,0]
        temp=temp.split("watch")
        url=temp[0]+"embed/"+temp[1][3:]+"?autoplay=1"
        new_df.iloc[i,0]=' <iframe src="{0}" width="853" height="480" frameborder="0" allowfullscreen></iframe>'.format(url)
    new_df= new_df.loc[:,['Link','Title','View Count','Likes','Dislikes','Sentiment']]
    print(new_df)
    pd.set_option('display.max_colwidth',-1)
    #new_df= new_df.reset_index()
    return render_template('table.html', data= new_df.to_html(escape=False))
#    return str(df)
@app.route("/youtube")
def yt_results():
    global yt_df
    return render_template('table.html', data= yt_df.to_html(escape=False))
    '''
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)
