
import sys

import os
from flask import Flask, render_template, request, send_file, logging
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

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


class AEmodel():
    def encoder(self):
        input_img = Input(shape=(190, 190, 1), name='input_layer')
        x = Conv2D(128, (3, 3), padding='same', name='block1_conv2')(input_img)
        x = BatchNormalization(name='block1_BN')(x)
        x = Activation('relu', name='block1_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        x = BatchNormalization(name='block2_BN')(x)
        x = Activation('relu', name='block2_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block2_pool')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
        x = BatchNormalization(name='block3_BN')(x)
        x = Activation('relu', name='block3_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block3_pool')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', name='block4_conv2')(x)
        x = BatchNormalization(name='block4_BN')(x)
        x = Activation('relu', name='block4_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block4_pool')(x)
        x = Dropout(0.25)(x)

        cx = GlobalAveragePooling2D(name='globalAve')(x)
        cx = Dropout(0.5)(cx)
        class_output = Dense(2, activation='softmax', name='class_output')(cx)

        x = UpSampling2D((2, 2), name='block7_upsample')(x)
        x = Conv2D(32, (3, 3), padding='same', name='block7_conv2')(x)
        x = BatchNormalization(name='block7_BN')(x)
        x = Activation('relu', name='block7_act')(x)
        x = Dropout(0.25)(x)

        x = UpSampling2D((2, 2), name='block8_upsample')(x)
        x = Conv2D(32, (3, 3), padding='same', name='block8_conv2')(x)
        x = BatchNormalization(name='block8_BN')(x)
        x = Activation('relu', name='block8_act')(x)
        x = Dropout(0.25)(x)

        x = UpSampling2D((2, 2), name='block9_upsample')(x)
        x = Conv2D(16, (3, 3), padding='same', name='block9_conv2')(x)
        x = BatchNormalization(name='block9_BN')(x)
        x = Activation('relu', name='block9_act')(x)
        x = Dropout(0.25)(x)

        x = UpSampling2D((2, 2), name='block10_upsample')(x)
        x = Conv2D(16, (3, 3), padding='same', name='block10_conv2')(x)
        x = BatchNormalization(name='block10_BN')(x)
        x = Activation('relu', name='block10_act')(x)
        x = Dropout(0.25)(x)
        decoded = Conv2D(1, (3, 3), name='decoder_output')(x)
        autoencoder = Model(inputs=input_img, outputs=[class_output, decoded])
        return autoencoder





if __name__ == "__main__":
    app.secret_key = 'key'
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True)
