import base64
import sys
import urllib

import os
from flask import Flask, render_template, request, send_file, logging
from io import BytesIO
import numpy as np
from PIL import Image
from skimage.io import imsave
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        return render_template('warning.html')





if __name__ == "__main__":
    app.secret_key = 'key'
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True)
