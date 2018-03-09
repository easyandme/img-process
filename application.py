import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import sys
import os
from flask import Flask, render_template, request, send_file, logging
from io import BytesIO
import numpy as np
from PIL import Image
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dropout, K
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
import pandas
from sklearn.manifold import TSNE

application = Flask(__name__)
application.logger.addHandler(logging.StreamHandler(sys.stdout))
application.logger.setLevel(logging.ERROR)


@application.route("/")
def index():
    return render_template("index.html")


@application.route("/about")
def about():
    return render_template("about.html")


@application.route('/render', methods=['GET', 'POST'])
def render():
    if request.method == 'POST':
        all_img_list = []
        label_num = len(request.files)
        label_values = []
        print(request.form)
        for i in range(1, label_num+1):
            input_name = "file" + str(i)
            label_name = "label" + str(i)
            if label_name in request.form.keys():
                label_value = request.form.get(label_name)
                label_values.append(label_value)
            if input_name in request.files:
                sub_img_list = request.files.getlist(input_name)
                all_img_list.append(sub_img_list)
        if all_img_list and all_img_list[0]:
            try:
                print(len(label_values))
                plot_url = feature_extract_TSNE(label_values, all_img_list)
                return render_template('result.html', png_url=plot_url)
            except Exception as err:
                if err:
                    return render_template('warning.html', err=err)
        else:
            return render_template('warning.html')


def feature_extract_TSNE(labels, all_img_list):
    allimages = []  # image files
    imglabel = []  # image labels
    num_label = 0
    for sub_img_list in all_img_list:
        sub_imgs = []
        for image in sub_img_list:
            img = Image.open(image)
            width, height = img.size
            if width != 190 or height != 190:
                img.thumbnail((190, 190), Image.ANTIALIAS)
            img = np.asarray(img, dtype=np.float32).reshape(190, 190, 1)
            img -= np.mean(img)
            sub_imgs.append(img)
        imglabel = np.concatenate((imglabel, np.ones((len(sub_imgs))) * num_label), axis=0)
        num_label += 1
        allimages.append(sub_imgs)

    allimages = np.asarray(allimages).reshape(-1, 190, 190, 1)
    imglabel = np.asarray(imglabel)

    autoencoder = ae_encoder()

    # Load weights
    autoencoder.load_weights('h5/ACbin_33x128fl128GA_weights.h5')
    autoencoder._make_predict_function()
    batch_size = 20

    # Extract output
    intermediate_layer_model = Model(inputs=autoencoder.input,
                                     outputs=autoencoder.get_layer('globalAve').output)

    intermediate_layer_model.compile('sgd', 'mse')
    # Output the latent layer
    intermediate_output = intermediate_layer_model.predict(
        allimages, batch_size=batch_size, verbose=1)

    K.clear_session()

    # TSNE
    Y0 = TSNE(n_components=2, init='random', random_state=0, perplexity=30,
              verbose=1).fit_transform(intermediate_output.reshape(intermediate_output.shape[0], -1))

    # Output scatter plot
    df = pandas.DataFrame(dict(x=Y0[:, 0], y=Y0[:, 1], label=imglabel))
    groups = df.groupby('label')

    # Plot grouped scatter
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    i = 0
    for name, group in groups:
        name = labels[i]
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=2, label=name, alpha=0.5)
        i += 1

    # Plot features
    plt.title('tSNE plot')
    ax.legend()

    # Encode, decode and output
    png = BytesIO()
    plt.savefig(png, format='png')
    png.seek(0)
    plot_url = base64.b64encode(png.getvalue()).decode()
    return plot_url


def ae_encoder():
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
    application.secret_key = 'key'
    port = int(os.environ.get("PORT", 5000))
    application.run(port=port, debug=False)
